/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using facebook::velox::test::BatchMaker;

namespace facebook::velox::exec::test {
namespace {

class LocalPartitionTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    window::prestosql::registerAllWindowFunctions();
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, vector_size_t size) {
    return makeFlatVector<T>(size, [start](auto row) { return start + row; });
  }

  template <typename T>
  FlatVectorPtr<T> makeFlatSequence(T start, T max, vector_size_t size) {
    return makeFlatVector<T>(
        size, [start, max](auto row) { return (start + row) % max; });
  }

  std::vector<std::shared_ptr<TempFilePath>> writeToFiles(
      const std::vector<RowVectorPtr>& vectors) {
    auto filePaths = makeFilePaths(vectors.size());
    for (auto i = 0; i < vectors.size(); i++) {
      writeToFile(filePaths[i]->getPath(), vectors[i]);
    }
    return filePaths;
  }

  void verifyExchangeSourceOperatorStats(
      const std::shared_ptr<exec::Task>& task,
      int expectedPositions,
      int expectedVectors,
      int expectedDrivers) {
    auto stats = task->taskStats().pipelineStats[0].operatorStats.front();
    ASSERT_EQ(stats.inputPositions, expectedPositions);
    ASSERT_EQ(stats.inputVectors, expectedVectors);
    ASSERT_EQ(stats.numDrivers, expectedDrivers);
    ASSERT_TRUE(stats.inputBytes > 0);

    ASSERT_EQ(stats.outputPositions, stats.inputPositions);
    ASSERT_EQ(stats.outputVectors, stats.inputVectors);
    ASSERT_EQ(stats.inputBytes, stats.outputBytes);
  }

  void assertTaskReferenceCount(
      const std::shared_ptr<exec::Task>& task,
      int expected) {
    // Make sure there is only one reference to Task left, i.e. no Driver is
    // blocked forever. Wait for a bit if that's not immediately the case.
    if (task.use_count() > expected) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    ASSERT_EQ(expected, task.use_count());
  }

  void waitForTaskCompletion(
      const std::shared_ptr<exec::Task>& task,
      exec::TaskState expected) {
    if (task->state() != expected) {
      auto& executor = folly::QueuedImmediateExecutor::instance();
      auto future = task->taskCompletionFuture()
                        .within(std::chrono::microseconds(1'000'000))
                        .via(&executor);
      future.wait();
      EXPECT_EQ(expected, task->state());
    }
  }
};

struct TestParam {
  uint32_t minLocalExchangePartitionCountToUsePartitionBuffer;
  uint32_t maxLocalExchangePartitionBufferSize;
  std::string_view name;
};

class LocalPartitionTestParametrized
    : public LocalPartitionTest,
      public testing::WithParamInterface<TestParam> {
 protected:
  void applyTestParameters(AssertQueryBuilder& queryBuilder) {
    const auto& params = GetParam();
    queryBuilder.config(
        core::QueryConfig::kMinLocalExchangePartitionCountToUsePartitionBuffer,
        std::to_string(
            params.minLocalExchangePartitionCountToUsePartitionBuffer));
    queryBuilder.config(
        core::QueryConfig::kMaxLocalExchangePartitionBufferSize,
        std::to_string(params.maxLocalExchangePartitionBufferSize));
  }
};

TEST_P(LocalPartitionTestParametrized, gather) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(-71, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto valuesNode = [&](int index) {
    return PlanBuilder(planNodeIdGenerator).values({vectors[index]}).planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {},
                    {
                        valuesNode(0),
                        valuesNode(1),
                        valuesNode(2),
                    })
                .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
                .planNode();

  auto task = assertQuery(op, "SELECT 300, -71, 152");
  verifyExchangeSourceOperatorStats(task, 300, 3, 1);

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  std::vector<core::PlanNodeId> scanNodeIds;

  auto tableScanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  op = PlanBuilder(planNodeIdGenerator)
           .localPartition(
               {},
               {
                   tableScanNode(),
                   tableScanNode(),
                   tableScanNode(),
               })
           .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
           .planNode();

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  applyTestParameters(queryBuilder);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  task = queryBuilder.assertResults("SELECT 300, -71, 152");

  verifyExchangeSourceOperatorStats(
      task,
      300,
      // no partition buffering for single output partition
      3,
      1);
}

TEST_F(LocalPartitionTest, gatherPreserveInputOrderWithSerialExecutionMode) {
  const std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatVector<int64_t>({10, 20})}),
      makeRowVector({makeFlatVector<int64_t>({30, 40})}),
      makeRowVector({makeFlatVector<int64_t>({50, 60})}),
      makeRowVector({makeFlatVector<int64_t>({70, 80})}),
      makeRowVector({makeFlatVector<int64_t>({90, 100})}),
      makeRowVector({makeFlatVector<int64_t>({110, 120})})};

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto valuesNode = [&](const std::vector<int>& indices) {
    std::vector<RowVectorPtr> values;
    for (const auto& index : indices) {
      values.push_back(vectors[index]);
    }
    return PlanBuilder(planNodeIdGenerator).values(values).planNode();
  };

  auto op =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {}, {valuesNode({0, 1, 2}), valuesNode({3}), valuesNode({4, 5})})
          .window({"row_number() over () as r"})
          .planNode();

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  queryBuilder.serialExecution(true).assertResults(
      "VALUES (10, 1), (20, 2), (30, 3), (40, 4), (50, 5), (60, 6), (70, 7), (80, 8), (90, 9), (100, 10), (110, 11), (120, 12)");
}

TEST_P(LocalPartitionTestParametrized, partition) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({makeFlatSequence<int32_t>(-71, 100)}),
  };

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;

  auto scanAggNode = [&]() {
    auto builder = PlanBuilder(planNodeIdGenerator);
    auto scanNode = builder.tableScan(rowType).planNode();
    scanNodeIds.push_back(scanNode->id());
    return builder.partialAggregation({"c0"}, {"count(1)"}).planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {
                        scanAggNode(),
                        scanAggNode(),
                        scanAggNode(),
                    })
                .partialAggregation({"c0"}, {"count(1)"})
                .planNode();

  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  applyTestParameters(queryBuilder);
  queryBuilder.maxDrivers(4);
  queryBuilder.config(core::QueryConfig::kMaxLocalExchangePartitionCount, "2");
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  auto task =
      queryBuilder.assertResults("SELECT c0, count(1) FROM tmp GROUP BY 1");

  verifyExchangeSourceOperatorStats(task, 300, 6, 2);
}

TEST_F(LocalPartitionTest, partitionBuffering) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({"c0"}, {makeFlatSequence<int32_t>(0, 100)}),
      makeRowVector({"c0"}, {makeFlatSequence<int32_t>(53, 100)}),
      makeRowVector({"c0"}, {makeFlatSequence<int32_t>(-71, 1000)}),
      makeRowVector({"c0"}, {makeFlatSequence<int32_t>(-69, 1000)}),
  };

  std::string query{"SELECT c0, count(1) FROM tmp GROUP BY 1"};
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"c0"},
              {PlanBuilder(planNodeIdGenerator).values(vectors).planNode()})
          .partialAggregation({"c0"}, {"count(1)"})
          .planNode();
  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(plan, duckDbQueryRunner_);
  queryBuilder.maxDrivers(2);

  std::unordered_map<std::string, std::string> configs;
  configs[core::QueryConfig::kMaxLocalExchangePartitionCount] = "2";

  // no buffer
  configs
      [core::QueryConfig::kMinLocalExchangePartitionCountToUsePartitionBuffer] =
          std::to_string(100);
  queryBuilder.configs(configs);
  verifyExchangeSourceOperatorStats(
      queryBuilder.assertResults(query), 2200, 8, 2);

  // tiny buffer
  configs
      [core::QueryConfig::kMinLocalExchangePartitionCountToUsePartitionBuffer] =
          std::to_string(2);
  configs[core::QueryConfig::kMaxLocalExchangePartitionBufferSize] =
      std::to_string(1);
  queryBuilder.configs(configs);
  verifyExchangeSourceOperatorStats(
      queryBuilder.assertResults(query), 2200, 8, 2);

  // small buffer
  configs[core::QueryConfig::kMaxLocalExchangePartitionBufferSize] =
      std::to_string(300);
  queryBuilder.configs(configs);
  verifyExchangeSourceOperatorStats(
      queryBuilder.assertResults(query), 2200, 6, 2);

  // medium buffer
  configs[core::QueryConfig::kMaxLocalExchangePartitionBufferSize] =
      std::to_string(1000);
  queryBuilder.configs(configs);
  verifyExchangeSourceOperatorStats(
      queryBuilder.assertResults(query), 2200, 4, 2);

  // large buffer
  configs[core::QueryConfig::kMaxLocalExchangePartitionBufferSize] =
      std::to_string(1000000);
  queryBuilder.configs(configs);
  verifyExchangeSourceOperatorStats(
      queryBuilder.assertResults(query), 2200, 2, 2);
}

TEST_F(LocalPartitionTest, partitionBufferingPreserveEncoding) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({"c0"}, {makeConstant<int32_t>(0, 100)}),
      makeRowVector({"c0"}, {makeConstant<int32_t>(0, 100)}),
      makeRowVector({"c0"}, {makeConstant<int32_t>(1, 100)}),
      makeRowVector({"c0"}, {makeConstant<int32_t>(1, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {"c0"},
              {PlanBuilder(planNodeIdGenerator).values(vectors).planNode()})
          .planNode();

  std::unordered_map<std::string, std::string> configs;
  // enable buffering
  configs
      [core::QueryConfig::kMinLocalExchangePartitionCountToUsePartitionBuffer] =
          std::to_string(2);
  configs[core::QueryConfig::kMaxLocalExchangePartitionBufferSize] =
      std::to_string(1000000);

  // enable preserve encoding
  configs[core::QueryConfig::kLocalExchangePartitionBufferPreserveEncoding] =
      "true";

  CursorParameters params;
  params.planNode = plan;
  params.copyResult = false;
  params.maxDrivers = 2;
  params.queryConfigs = configs;
  auto cursor = TaskCursor::create(params);
  int numRows = 0;
  int numVectors = 0;
  while (cursor->moveNext()) {
    auto* batch = cursor->current()->as<RowVector>();
    ASSERT_EQ(batch->childrenSize(), 1);
    auto& column = batch->childAt(0);
    ASSERT_EQ(column->encoding(), VectorEncoding::Simple::CONSTANT);
    numRows += batch->size();
    numVectors++;
  }
  ASSERT_EQ(numRows, 400);
  ASSERT_EQ(numVectors, 2);
}

TEST_F(LocalPartitionTest, maxBufferSizeGather) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  auto valuesNode = [&](int start, int end) {
    return PlanBuilder(planNodeIdGenerator)
        .values(std::vector<RowVectorPtr>(
            vectors.begin() + start, vectors.begin() + end))
        .planNode();
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {},
                    {
                        valuesNode(0, 7),
                        valuesNode(7, 14),
                        valuesNode(14, 21),
                    })
                .singleAggregation({}, {"count(1)", "min(c0)", "max(c0)"})
                .planNode();

  auto task = AssertQueryBuilder(op, duckDbQueryRunner_)
                  .config(core::QueryConfig::kMaxLocalExchangeBufferSize, "100")
                  .assertResults("SELECT 2100, -71, 228");

  verifyExchangeSourceOperatorStats(task, 2100, 21, 1);
}

TEST_F(LocalPartitionTest, maxBufferSizePartition) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }

  createDuckDbTable(vectors);

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;

  auto scanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {
                        scanNode(),
                        scanNode(),
                        scanNode(),
                    })
                .partialAggregation({"c0"}, {"count(1)"})
                .planNode();

  auto makeQueryBuilder = [&](const char* bufferSize) {
    AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);

    queryBuilder.maxDrivers(2);
    for (auto i = 0; i < filePaths.size(); ++i) {
      queryBuilder.split(
          scanNodeIds[i % 3], makeHiveConnectorSplit(filePaths[i]->getPath()));
    }
    queryBuilder.config(
        core::QueryConfig::kMaxLocalExchangeBufferSize, bufferSize);
    return queryBuilder;
  };

  // Set an artificially low buffer size limit to trigger blocking behavior.
  auto task = makeQueryBuilder("100").assertResults(
      "SELECT c0, count(1) FROM tmp GROUP BY 1");
  verifyExchangeSourceOperatorStats(task, 2100, 42, 2);

  // Re-run with higher memory limit (enough to hold ~10 vectors at a time).
  task = makeQueryBuilder("10240").assertResults(
      "SELECT c0, count(1) FROM tmp GROUP BY 1");
  verifyExchangeSourceOperatorStats(task, 2100, 42, 2);
}

TEST_F(LocalPartitionTest, indicesBufferCapacity) {
  std::vector<RowVectorPtr> vectors;
  for (auto i = 0; i < 21; i++) {
    vectors.emplace_back(makeRowVector({makeFlatVector<int32_t>(
        100, [i](auto row) { return -71 + i * 10 + row; })}));
  }
  auto filePaths = writeToFiles(vectors);
  auto rowType = asRowType(vectors[0]->type());
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodeId> scanNodeIds;
  auto scanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };
  CursorParameters params;
  params.planNode = PlanBuilder(planNodeIdGenerator)
                        .localPartition(
                            {"c0"},
                            {
                                scanNode(),
                                scanNode(),
                                scanNode(),
                            })
                        .planNode();
  params.copyResult = false;
  params.maxDrivers = 2;
  auto cursor = TaskCursor::create(params);
  for (auto i = 0; i < filePaths.size(); ++i) {
    auto id = scanNodeIds[i % 3];
    cursor->task()->addSplit(
        id, Split(makeHiveConnectorSplit(filePaths[i]->getPath())));
    cursor->task()->noMoreSplits(id);
  }
  int numRows = 0;
  int capacity = 0;
  while (cursor->moveNext()) {
    auto* batch = cursor->current()->as<RowVector>();
    ASSERT_EQ(batch->childrenSize(), 1);
    auto& column = batch->childAt(0);
    ASSERT_EQ(column->encoding(), VectorEncoding::Simple::DICTIONARY);
    numRows += batch->size();
    capacity += column->wrapInfo()->capacity();
  }
  ASSERT_EQ(numRows, 2100);
  // MemoryPool::preferredSize is capped at 1.5 times the requested size.
  ASSERT_LE(capacity, 1.5 * numRows * sizeof(vector_size_t));
}

TEST_F(LocalPartitionTest, blockingOnLocalExchangeQueue) {
  auto localExchangeBufferSize = "1024";
  auto baseVector = vectorMaker_.flatVector<int64_t>(
      10240, [](auto row) { return row / 10; });
  // Make a small flat vector of one row and roughly 8 bytes that is
  // smaller than the localExchangeBufferSize.
  auto smallInput = vectorMaker_.rowVector(
      {"c0"}, {makeFlatVector<int64_t>(1, folly::identity)});
  // Make a small dictionary vector of one row with a base vector larger than
  // the localExchangeBufferSize.
  auto dictionaryInput = vectorMaker_.rowVector(
      {"c0"}, {wrapInDictionary(makeIndices({0}), baseVector)});
  // Make a large dictionary vector of 1024 rows and roughly 8KB that is larger
  // than the localExchangeBufferSize.
  auto largeInput = vectorMaker_.rowVector(
      {"c0"},
      {wrapInDictionary(
          makeIndices(baseVector->size(), [](auto row) { return row; }),
          baseVector)});

  struct {
    RowVectorPtr input;
    int64_t numBlocked;

    std::string debugString() const {
      return fmt::format(
          "inputBatchBytes: {}, numBlocked: {}",
          input->estimateFlatSize(),
          numBlocked);
    }
  } testSettings[] = {
      {smallInput, 0}, // Small input will not make LocalPartition blocked.
      {dictionaryInput, 1}, // Large dictiionary values will make LocalPartition
                            // blocked.
      {largeInput, 1}}; // Large input will make LocalPartition blocked.

  for (const auto& test : testSettings) {
    SCOPED_TRACE(test.debugString());

    createDuckDbTable({test.input});

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    core::PlanNodeId nodeId;
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .localPartition(
                        {"c0"},
                        {PlanBuilder(planNodeIdGenerator)
                             .values({test.input})
                             .planNode()})
                    .capturePlanNodeId(nodeId)
                    .singleAggregation({"c0"}, {"count(1)"})
                    .planNode();
    auto task = AssertQueryBuilder(duckDbQueryRunner_)
                    .plan(plan)
                    .maxDrivers(4)
                    .config(
                        core::QueryConfig::kMaxLocalExchangeBufferSize,
                        localExchangeBufferSize)
                    .assertResults("SELECT c0, count(1) FROM tmp GROUP BY c0");
    ASSERT_EQ(
        exec::toPlanStats(task->taskStats())
            .at(nodeId)
            .customStats["blockedWaitForConsumerTimes"]
            .sum,
        test.numBlocked);
  }
}

TEST_P(LocalPartitionTestParametrized, multipleExchanges) {
  std::vector<RowVectorPtr> vectors = {
      makeRowVector({
          makeFlatSequence<int32_t>(0, 100),
          makeFlatSequence<int64_t>(0, 7, 100),
      }),
      makeRowVector({
          makeFlatSequence<int32_t>(53, 100),
          makeFlatSequence<int64_t>(0, 11, 100),
      }),
      makeRowVector({
          makeFlatSequence<int32_t>(-71, 100),
          makeFlatSequence<int64_t>(0, 13, 100),
      }),
  };

  auto filePaths = writeToFiles(vectors);

  auto rowType = asRowType(vectors[0]->type());

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  std::vector<core::PlanNodeId> scanNodeIds;

  auto tableScanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  // Make a plan with 2 local exchanges. UNION ALL results of 3 table scans.
  // Group by 0, 1 and compute counts. Group by 0 and compute counts and sums.
  // First exchange re-partitions the results of table scan on two keys. Second
  // exchange re-partitions the results on just the first key.
  auto op = PlanBuilder(planNodeIdGenerator)
                .localPartition(
                    {"c0"},
                    {PlanBuilder(planNodeIdGenerator)
                         .localPartition(
                             {"c0", "c1"},
                             {
                                 tableScanNode(),
                                 tableScanNode(),
                                 tableScanNode(),
                             })
                         .partialAggregation({"c0", "c1"}, {"count(1)"})
                         .planNode()})
                .partialAggregation({"c0"}, {"count(1)", "sum(a0)"})
                .planNode();

  createDuckDbTable(vectors);

  AssertQueryBuilder queryBuilder(op, duckDbQueryRunner_);
  applyTestParameters(queryBuilder);
  for (auto i = 0; i < filePaths.size(); ++i) {
    queryBuilder.split(
        scanNodeIds[i], makeHiveConnectorSplit(filePaths[i]->getPath()));
  }

  queryBuilder.maxDrivers(2).assertResults(
      "SELECT c0, count(1), sum(cnt) FROM ("
      "   SELECT c0, c1, count(1) as cnt FROM tmp GROUP BY 1, 2"
      ") t GROUP BY 1");
}

TEST_P(LocalPartitionTestParametrized, earlyCompletion) {
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatSequence(3, 100)}),
      makeRowVector({makeFlatSequence(7, 100)}),
      makeRowVector({makeFlatSequence(11, 100)}),
      makeRowVector({makeFlatSequence(13, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {}, {PlanBuilder(planNodeIdGenerator).values(data).planNode()})
          .limit(0, 2, true)
          .planNode();

  AssertQueryBuilder queryBuilder(plan, duckDbQueryRunner_);
  applyTestParameters(queryBuilder);
  auto task = queryBuilder.maxDrivers(2).assertResults("VALUES (3), (4)");

  verifyExchangeSourceOperatorStats(task, 100, 1, 1);
  // Make sure there is only one reference to Task left, i.e. no Driver is
  // blocked forever.
  assertTaskReferenceCount(task, 1);
}

TEST_F(LocalPartitionTest, earlyCancelation) {
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatSequence(3, 100)}),
      makeRowVector({makeFlatSequence(7, 100)}),
      makeRowVector({makeFlatSequence(11, 100)}),
      makeRowVector({makeFlatSequence(13, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .localPartition(
              {}, {PlanBuilder(planNodeIdGenerator).values(data).planNode()})
          .limit(0, 2'000, true)
          .planNode();

  CursorParameters params;
  params.planNode = plan;
  // Make sure results are queued one batch at a time.
  params.bufferedBytes = 100;

  auto cursor = TaskCursor::create(params);
  const auto& task = cursor->task();

  // Fetch first batch of data.
  ASSERT_TRUE(cursor->moveNext());
  ASSERT_EQ(100, cursor->current()->size());

  // Cancel the task.
  task->requestCancel();

  // Fetch the remaining results. This will throw since only one vector can be
  // buffered in the cursor.
  try {
    while (cursor->moveNext()) {
      ;
      FAIL() << "Expected a throw due to cancellation";
    }
  } catch (const std::exception&) {
  }

  // Wait for task to transition to final state.
  waitForTaskCompletion(task, exec::TaskState::kCanceled);

  // Make sure there is only one reference to Task left, i.e. no Driver is
  // blocked forever.
  assertTaskReferenceCount(task, 1);
}

TEST_F(LocalPartitionTest, producerError) {
  std::vector<RowVectorPtr> data = {
      makeRowVector({makeFlatSequence(3, 100)}),
      makeRowVector({makeFlatSequence(7, 100)}),
      makeRowVector({makeFlatSequence(-11, 100)}),
      makeRowVector({makeFlatSequence(-13, 100)}),
  };

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .values(data)
                           .project({"7 / c0"})
                           .planNode()})
                  .limit(0, 2'000, true)
                  .planNode();

  CursorParameters params;
  params.planNode = plan;

  auto cursor = TaskCursor::create(params);
  const auto& task = cursor->task();

  // Expect division by zero error.
  ASSERT_THROW(while (cursor->moveNext()) { ; }, VeloxException);

  // Wait for task to transition to failed state.
  waitForTaskCompletion(task, exec::TaskState::kFailed);

  // Make sure there is only one reference to Task left, i.e. no Driver is
  // blocked forever.
  assertTaskReferenceCount(task, 1);
}

TEST_F(LocalPartitionTest, unionAll) {
  auto data1 = makeRowVector(
      {"d0", "d1"},
      {makeFlatVector<int32_t>({10, 11}),
       makeFlatVector<StringView>({"x", "y"})});
  auto data2 = makeRowVector(
      {"e0", "e1"},
      {makeFlatVector<int32_t>({20, 21}),
       makeFlatVector<StringView>({"z", "w"})});

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {PlanBuilder(planNodeIdGenerator)
                           .values({data1})
                           .project({"d0 as c0", "d1 as c1"})
                           .planNode(),
                       PlanBuilder(planNodeIdGenerator)
                           .values({data2})
                           .project({"e0 as c0", "e1 as c1"})
                           .planNode()})
                  .planNode();

  assertQuery(
      plan,
      "WITH t1 AS (VALUES (10, 'x'), (11, 'y')), "
      "t2 AS (VALUES (20, 'z'), (21, 'w')) "
      "SELECT * FROM t1 UNION ALL SELECT * FROM t2");
}

TEST_P(LocalPartitionTestParametrized, unionAllLocalExchange) {
  auto data1 = makeRowVector({"d0"}, {makeFlatVector<StringView>({"x"})});
  auto data2 = makeRowVector({"e0"}, {makeFlatVector<StringView>({"y"})});
  for (bool serialExecutionMode : {false, true}) {
    SCOPED_TRACE(fmt::format("serialExecutionMode {}", serialExecutionMode));
    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    AssertQueryBuilder queryBuilder(duckDbQueryRunner_);
    applyTestParameters(queryBuilder);
    queryBuilder.serialExecution(serialExecutionMode)
        .plan(PlanBuilder(planNodeIdGenerator)
                  .localPartitionRoundRobin(
                      {PlanBuilder(planNodeIdGenerator)
                           .values({data1})
                           .project({"d0 as c0"})
                           .planNode(),
                       PlanBuilder(planNodeIdGenerator)
                           .values({data2})
                           .project({"e0 as c0"})
                           .planNode()})
                  .project({"length(c0)"})
                  .planNode())
        .assertResults(
            "SELECT length(c0) FROM ("
            "   SELECT * FROM (VALUES ('x')) as t1(c0) UNION ALL "
            "   SELECT * FROM (VALUES ('y')) as t2(c0)"
            ")");
  }
}

namespace {
using BlockingCallback = std::function<BlockingReason(ContinueFuture*)>;
using FinishCallback = std::function<void(bool)>;

class BlockingNode : public core::PlanNode {
 public:
  BlockingNode(const core::PlanNodeId& id, const core::PlanNodePtr& input)
      : PlanNode(id), sources_{input} {}

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }

  const std::vector<std::shared_ptr<const PlanNode>>& sources() const override {
    return sources_;
  }

  std::string_view name() const override {
    return "BlockingNode";
  }

 private:
  void addDetails(std::stringstream& /* stream */) const override {}
  std::vector<core::PlanNodePtr> sources_;
};

class BlockingOperator : public Operator {
 public:
  BlockingOperator(
      DriverCtx* ctx,
      int32_t id,
      const std::shared_ptr<const BlockingNode>& node,
      const BlockingCallback& blockingCallback,
      const FinishCallback& finishCallback)
      : Operator(ctx, node->outputType(), id, node->id(), "BlockedNoFuture"),
        blockingCallback_(blockingCallback),
        finishCallback_(finishCallback) {}

  bool needsInput() const override {
    return !noMoreInput_ && !input_;
  }

  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }

  RowVectorPtr getOutput() override {
    return std::move(input_);
  }

  bool isFinished() override {
    const bool finished = noMoreInput_ && input_ == nullptr;
    finishCallback_(finished);
    return finished;
  }

  BlockingReason isBlocked(ContinueFuture* future) override {
    return blockingCallback_(future);
  }

 private:
  const BlockingCallback blockingCallback_;
  const FinishCallback finishCallback_;
};

class BlockingNodeFactory : public Operator::PlanNodeTranslator {
 public:
  explicit BlockingNodeFactory(
      const BlockingCallback& blockingCallback,
      const FinishCallback& finishCallback)
      : blockingCallback_(blockingCallback), finishCallback_(finishCallback) {}

  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    auto blockingNode = std::dynamic_pointer_cast<const BlockingNode>(node);
    if (blockingNode == nullptr) {
      return nullptr;
    }
    return std::make_unique<BlockingOperator>(
        ctx, id, blockingNode, blockingCallback_, finishCallback_);
  }

  std::optional<uint32_t> maxDrivers(
      const core::PlanNodePtr& /*unused*/) override {
    return std::numeric_limits<uint32_t>::max();
  }

 private:
  const BlockingCallback blockingCallback_;
  const FinishCallback finishCallback_;
};
} // namespace

TEST_F(LocalPartitionTest, unionAllLocalExchangeWithInterDependency) {
  const auto data1 = makeRowVector({"d0"}, {makeFlatVector<StringView>({"x"})});
  const auto data2 = makeRowVector({"e0"}, {makeFlatVector<StringView>({"y"})});

  for (bool serialExecutionMode : {false, true}) {
    SCOPED_TRACE(fmt::format("serialExecutionMode {}", serialExecutionMode));
    Operator::unregisterAllOperators();

    std::mutex mutex;
    std::vector<ContinuePromise> promises;
    promises.reserve(2);
    std::vector<ContinueFuture> futures;
    futures.reserve(2);
    for (int i = 0; i < 2; ++i) {
      auto [blockPromise, blockFuture] = makeVeloxContinuePromiseContract(
          "unionAllLocalExchangeWithInterDependency");
      promises.push_back(std::move(blockPromise));
      futures.push_back(std::move(blockFuture));
    }

    std::atomic_uint32_t numBlocks{0};
    auto blockingCallback = [&](ContinueFuture* future) -> BlockingReason {
      std::lock_guard<std::mutex> l(mutex);
      if (numBlocks >= 2) {
        return BlockingReason::kNotBlocked;
      }
      *future = std::move(futures[numBlocks]);
      ++numBlocks;
      return BlockingReason::kWaitForConsumer;
    };

    auto finishCallback = [&](bool finished) {
      if (!finished) {
        return;
      }
      std::lock_guard<std::mutex> l(mutex);
      for (auto& promise : promises) {
        if (!promise.isFulfilled()) {
          promise.setValue();
        }
      }
    };

    Operator::registerOperator(std::make_unique<BlockingNodeFactory>(
        std::move(blockingCallback), std::move(finishCallback)));

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan = PlanBuilder(planNodeIdGenerator)
                    .localPartitionRoundRobin(
                        {PlanBuilder(planNodeIdGenerator)
                             .values({data1})
                             .project({"d0 as c0"})
                             .addNode([](const core::PlanNodeId& id,
                                         const core::PlanNodePtr& input) {
                               return std::make_shared<BlockingNode>(id, input);
                             })
                             .planNode(),
                         PlanBuilder(planNodeIdGenerator)
                             .values({data2})
                             .project({"e0 as c0"})
                             .addNode([](const core::PlanNodeId& id,
                                         const core::PlanNodePtr& input) {
                               return std::make_shared<BlockingNode>(id, input);
                             })
                             .planNode()})
                    .project({"length(c0)"})
                    .planNode();

    auto thread = std::thread([&]() {
      AssertQueryBuilder(duckDbQueryRunner_)
          .serialExecution(serialExecutionMode)
          .plan(std::move(plan))
          .assertResults(
              "SELECT length(c0) FROM ("
              "   SELECT * FROM (VALUES ('x')) as t1(c0) UNION ALL "
              "   SELECT * FROM (VALUES ('y')) as t2(c0)"
              ")");
    });

    while (numBlocks != 2) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10)); // NOLINT
    }
    promises[0].setValue();

    thread.join();
  }
}

TEST_F(
    LocalPartitionTest,
    taskErrorWithBlockedDriverFutureUnderSerializedExecutionMode) {
  const auto data1 = makeRowVector({"d0"}, {makeFlatVector<StringView>({"x"})});
  const auto data2 = makeRowVector({"e0"}, {makeFlatVector<StringView>({"y"})});

  Operator::unregisterAllOperators();

  std::mutex mutex;
  auto contract = makeVeloxContinuePromiseContract(
      "driverFutureErrorUnderSerializedExecutionMode");

  std::atomic_uint32_t numBlocks{0};
  auto blockingCallback = [&](ContinueFuture* future) -> BlockingReason {
    std::lock_guard<std::mutex> l(mutex);
    if (numBlocks++ > 0) {
      return BlockingReason::kNotBlocked;
    }
    *future = std::move(contract.second);
    return BlockingReason::kWaitForConsumer;
  };

  auto finishCallback = [&](bool /*unused*/) {};

  Operator::registerOperator(std::make_unique<BlockingNodeFactory>(
      std::move(blockingCallback), std::move(finishCallback)));

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartitionRoundRobin(
                      {PlanBuilder(planNodeIdGenerator)
                           .values({data1})
                           .project({"d0 as c0"})
                           .addNode([](const core::PlanNodeId& id,
                                       const core::PlanNodePtr& input) {
                             return std::make_shared<BlockingNode>(id, input);
                           })
                           .planNode(),
                       PlanBuilder(planNodeIdGenerator)
                           .values({data2})
                           .project({"e0 as c0"})
                           .addNode([](const core::PlanNodeId& id,
                                       const core::PlanNodePtr& input) {
                             return std::make_shared<BlockingNode>(id, input);
                           })
                           .planNode()})
                  .project({"length(c0)"})
                  .planNode();

  auto thread = std::thread([&]() {
    VELOX_ASSERT_THROW(
        AssertQueryBuilder(duckDbQueryRunner_)
            .serialExecution(true)
            .plan(std::move(plan))
            .assertResults(
                "SELECT length(c0) FROM ("
                "   SELECT * FROM (VALUES ('x')) as t1(c0) UNION ALL "
                "   SELECT * FROM (VALUES ('y')) as t2(c0)"
                ")"),
        "");
  });

  while (numBlocks < 2) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // NOLINT
  }
  std::this_thread::sleep_for(std::chrono::seconds(1)); // NOLINT

  auto tasks = Task::getRunningTasks();
  ASSERT_EQ(tasks.size(), 1);
  tasks[0]->requestAbort().wait();
  thread.join();
}

TEST_F(LocalPartitionTest, vectorPool) {
  LocalExchangeVectorPool vectorPool(10);
  std::vector<RowVector*> vectors;
  auto makeVector = [&] {
    auto vector =
        BaseVector::create<RowVector>(ROW({"c0"}, {BIGINT()}), 1, pool());
    vectors.push_back(vector.get());
    return vector;
  };
  vectorPool.push(makeVector(), 5);
  auto multiReferenced = makeVector();
  vectorPool.push(multiReferenced, 2);
  vectorPool.push(makeVector(), 3);
  vectorPool.push(makeVector(), 1);
  auto vector = vectorPool.pop();
  ASSERT_TRUE(vector != nullptr);
  ASSERT_EQ(vector.get(), vectors[0]);
  vector = vectorPool.pop();
  ASSERT_TRUE(vector != nullptr);
  ASSERT_EQ(vector.get(), vectors[2]);
  ASSERT_FALSE(vectorPool.pop());
}

TEST_F(LocalPartitionTest, barrier) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  std::vector<RowVectorPtr> vectors;
  const int numSources{3};
  std::vector<std::vector<std::shared_ptr<TempFilePath>>> tempFiles(numSources);
  const int numSplits{5};
  for (int i = 0; i < numSources; ++i) {
    std::vector<RowVectorPtr> sourceVectors;
    for (int32_t j = 0; j < numSplits; ++j) {
      auto vector = std::dynamic_pointer_cast<RowVector>(
          BatchMaker::createBatch(rowType, 100, *pool_));
      sourceVectors.push_back(vector);
      tempFiles[i].push_back(TempFilePath::create());
    }
    HiveConnectorTestBase::writeToFiles(
        toFilePaths(tempFiles[i]), sourceVectors);
    std::copy(
        sourceVectors.begin(),
        sourceVectors.end(),
        std::back_inserter(vectors));
  }
  createDuckDbTable(vectors);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  std::vector<core::PlanNodeId> scanNodeIds;
  auto tableScanNode = [&]() {
    auto node = PlanBuilder(planNodeIdGenerator).tableScan(rowType).planNode();
    scanNodeIds.push_back(node->id());
    return node;
  };

  auto plan = PlanBuilder(planNodeIdGenerator)
                  .localPartition(
                      {},
                      {
                          tableScanNode(),
                          tableScanNode(),
                          tableScanNode(),
                      })
                  .planNode();

  for (const auto hasBarrier : {false, true}) {
    SCOPED_TRACE(fmt::format("hasBarrier {}", hasBarrier));
    AssertQueryBuilder queryBuilder(plan, duckDbQueryRunner_);
    queryBuilder.barrierExecution(hasBarrier).serialExecution(true);
    for (auto i = 0; i < numSources; ++i) {
      for (auto j = 0; j < numSplits; ++j) {
        queryBuilder.split(
            scanNodeIds[i], makeHiveConnectorSplit(tempFiles[i][j]->getPath()));
      }
    }

    const auto task = queryBuilder.assertResults("SELECT * FROM tmp");
    ASSERT_EQ(task->taskStats().numBarriers, hasBarrier ? numSplits : 0);
  }
}

INSTANTIATE_TEST_SUITE_P(
    LocalExchangePartitionBuffer,
    LocalPartitionTestParametrized,
    testing::Values(
        TestParam{1000, 1000, "partition_buffer_disabled"},
        TestParam{0, 1024, "partition_buffer_enabled"},
        TestParam{0, 0, "partition_buffer_enabled_always_flush"},
        TestParam{0, 10 * 1024 * 1024, "partition_buffer_enabled_never_flush"}),
    [](const testing::TestParamInfo<TestParam>& info) {
      return std::string{info.param.name};
    });

} // namespace
} // namespace facebook::velox::exec::test
