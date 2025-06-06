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
#pragma once

#include "velox/exec/Operator.h"
#include "velox/exec/VectorHasher.h"

namespace facebook::velox::exec {

/// Keeps track of the total size in bytes of the data buffered in all
/// LocalExchangeQueues.
class LocalExchangeMemoryManager {
 public:
  explicit LocalExchangeMemoryManager(int64_t maxBufferSize)
      : maxBufferSize_{maxBufferSize} {}

  /// Returns 'true' if memory limit is reached or exceeded and sets future that
  /// will be complete when memory usage is update to be below the limit.
  bool increaseMemoryUsage(ContinueFuture* future, int64_t added);

  /// Decreases the memory usage by 'removed' bytes. If the memory usage goes
  /// below the limit after the decrease, the function returns 'promises_' to
  /// caller to fulfill.
  std::vector<ContinuePromise> decreaseMemoryUsage(int64_t removed);

  /// Returns the maximum buffer size in bytes.
  int64_t maxBufferBytes() const {
    return maxBufferSize_;
  }

  /// Returns the current buffer size in bytes.
  int64_t bufferedBytes() const {
    return bufferedBytes_;
  }

 private:
  const int64_t maxBufferSize_;
  std::mutex mutex_;
  tsan_atomic<int64_t> bufferedBytes_{0};
  std::vector<ContinuePromise> promises_;
};

/// A vector pool to reuse the RowVector and DictionaryVectors.  Only
/// exclusively owned vectors will be reused.
class LocalExchangeVectorPool {
 public:
  explicit LocalExchangeVectorPool(int64_t capacity) : capacity_(capacity) {}

  /// `size' is the estimated size of the `vector' (e.g. taking shared
  /// dictionary into consideration).
  void push(const RowVectorPtr& vector, int64_t size);

  RowVectorPtr pop();

 private:
  const int64_t capacity_;
  int64_t totalSize_{0};
  folly::Synchronized<std::queue<std::pair<RowVectorPtr, int64_t>>> pool_;
};

/// Buffers data for a single partition produced by local exchange. Allows
/// multiple producers to enqueue data and multiple consumers fetch data. Each
/// producer must be registered with a call to 'addProducer'. 'noMoreProducers'
/// must be called after all producers have been registered. A producer calls
/// 'enqueue' multiple time to put the data and calls 'noMoreData' when done.
/// Consumers call 'next' repeatedly to fetch the data.
class LocalExchangeQueue {
 public:
  LocalExchangeQueue(
      std::shared_ptr<LocalExchangeMemoryManager> memoryManager,
      std::shared_ptr<LocalExchangeVectorPool> vectorPool,
      int partition)
      : memoryManager_{std::move(memoryManager)},
        vectorPool_{std::move(vectorPool)},
        partition_{partition} {}

  std::string toString() const {
    return fmt::format("LocalExchangeQueue({})", partition_);
  }

  void addProducer();

  void noMoreProducers();

  /// Used by a producer to add data. Returning kNotBlocked if can accept more
  /// data. Otherwise returns kWaitForConsumer and sets future that will be
  /// completed when ready to accept more data.
  BlockingReason
  enqueue(RowVectorPtr input, int64_t inputBytes, ContinueFuture* future);

  /// Called by a producer to indicate the producer pipeline has been drained
  /// under barrier processing.
  void drain();

  /// Called by a producer to indicate that no more data will be added.
  void noMoreData();

  /// Used by a consumer to fetch some data. Returns kNotBlocked and sets data
  /// to nullptr if all data has been fetched and all producers are done
  /// producing data. Returns kWaitForProducer if there is no data, but some
  /// producers are not done producing data. Sets future that will be completed
  /// once there is data to fetch or if all producers report completion.
  ///
  /// @param pool Memory pool used to copy the data before returning.
  /// @param drained Set to true if all the producers of this queue have been
  /// drained under barrier processing.
  BlockingReason next(
      ContinueFuture* future,
      memory::MemoryPool* pool,
      RowVectorPtr* data,
      bool& drained);

  bool isFinished();

  /// Drop remaining data from the queue and notify consumers and producers if
  /// called before all the data has been processed. No-op otherwise.
  void close();

  /// Get a reusable vector from the vector pool.  Return nullptr if none is
  /// available.
  RowVectorPtr getVector() {
    return vectorPool_->pop();
  }

  /// Returns true if all producers have sent no more data signal.
  bool testingProducersDone() const;

 private:
  using Queue = std::queue<std::pair<RowVectorPtr, int64_t>>;

  bool isFinishedLocked(const Queue& queue) const;

  bool testAndClearDrainedLocked();

  const std::shared_ptr<LocalExchangeMemoryManager> memoryManager_;
  const std::shared_ptr<LocalExchangeVectorPool> vectorPool_;
  const int partition_;

  folly::Synchronized<Queue> queue_;
  // Satisfied when data becomes available or all producers report that they
  // finished producing, e.g. queue_ is not empty or noMoreProducers_ is true
  // and pendingProducers_ is zero.
  std::vector<ContinuePromise> consumerPromises_;
  int pendingProducers_{0};
  bool noMoreProducers_{false};
  // The number of drained producers when the task is under barrier processing.
  // If it equals to 'pendingProducers_', then the queue is drained. The
  // consumer receives the drained signal on the next call to 'next', and
  // 'drainedProducers_' is reset to zero.
  int drainedProducers_{0};
  bool closed_{false};
};

/// Fetches data for a single partition produced by local exchange from
/// LocalExchangeQueue.
class LocalExchange : public SourceOperator {
 public:
  LocalExchange(
      int32_t operatorId,
      DriverCtx* ctx,
      RowTypePtr outputType,
      const std::string& planNodeId,
      int partition);

  std::string toString() const override {
    return fmt::format("LocalExchange({})", partition_);
  }

  bool startDrain() override {
    return false;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  RowVectorPtr getOutput() override;

  bool isFinished() override;

  /// Close exchange queue. If called before all data has been processed,
  /// notifies the producer that no more data is needed.
  void close() override;

 private:
  const int partition_;
  const std::shared_ptr<LocalExchangeQueue> queue_{nullptr};
  ContinueFuture future_;
  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
};

/// Hash partitions the data using specified keys. The number of partitions is
/// determined by the number of LocalExchangeQueues(s) found in the task.
class LocalPartition : public Operator {
 public:
  LocalPartition(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::LocalPartitionNode>& planNode,
      bool eagerFlush);

  std::string toString() const override {
    return fmt::format("LocalPartition({})", numPartitions_);
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  /// Always true but the caller will check isBlocked before adding input, hence
  /// the blocked state does not accumulate input.
  bool needsInput() const override {
    return true;
  }

  bool startDrain() override {
    VELOX_CHECK(isDraining());
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  void noMoreInput() override;

  bool isFinished() override;

 protected:
  void prepareForInput(RowVectorPtr& input);

  void allocateIndexBuffers(const std::vector<vector_size_t>& sizes);

  RowVectorPtr processPartition(
      const RowVectorPtr& input,
      vector_size_t size,
      int partition,
      const BufferPtr& indices,
      const vector_size_t* rawIndices);

  const std::vector<std::shared_ptr<LocalExchangeQueue>> queues_;
  const size_t numPartitions_;
  std::unique_ptr<core::PartitionFunction> partitionFunction_;

  std::vector<BlockingReason> blockingReasons_;
  std::vector<ContinueFuture> futures_;

  /// Reusable memory for hash calculation.
  std::vector<uint32_t> partitions_;
  /// Reusable buffers for input partitioning.
  std::vector<BufferPtr> indexBuffers_;
  std::vector<vector_size_t*> rawIndices_;

 private:
  RowVectorPtr wrapChildren(
      const RowVectorPtr& input,
      vector_size_t size,
      const BufferPtr& indices,
      RowVectorPtr reusable);

  void copy(
      const RowVectorPtr& input,
      const folly::Range<const BaseVector::CopyRange*>& ranges,
      VectorPtr& target);

  const uint64_t singlePartitionBufferSize_;
  std::vector<BaseVector::CopyRange> copyRanges_;
  std::vector<VectorPtr> partitionBuffers_;
  const bool partitionBufferPreserveEncoding_;
};

} // namespace facebook::velox::exec
