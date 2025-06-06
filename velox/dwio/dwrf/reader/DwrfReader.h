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

#include "folly/Executor.h"
#include "folly/synchronization/Baton.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/UnitLoader.h"
#include "velox/dwio/dwrf/reader/SelectiveDwrfReader.h"

namespace facebook::velox::dwrf {

class ColumnReader;
class DwrfUnit;

class DwrfOptions : public dwio::common::FormatSpecificOptions {
 public:
  void setColumnReaderFactory(
      std::shared_ptr<ColumnReaderFactory> columnReaderFactory) {
    columnReaderFactory_ = std::move(columnReaderFactory);
  }

  const std::shared_ptr<ColumnReaderFactory>& columnReaderFactory() const {
    return columnReaderFactory_;
  }

 private:
  std::shared_ptr<ColumnReaderFactory> columnReaderFactory_;
};

class DwrfRowReader : public StrideIndexProvider,
                      public StripeReaderBase,
                      public dwio::common::RowReader {
 public:
  /**
   * Constructor that lets the user specify additional options.
   * @param reader contents of the file
   * @param options options for reading
   */
  DwrfRowReader(
      const std::shared_ptr<ReaderBase>& reader,
      const dwio::common::RowReaderOptions& options);

  ~DwrfRowReader() override = default;

  // Select the columns from the options object
  const dwio::common::ColumnSelector& getColumnSelector() const {
    return *columnSelector_;
  }

  const std::shared_ptr<dwio::common::ColumnSelector>& getColumnSelectorPtr()
      const {
    return columnSelector_;
  }

  const dwio::common::RowReaderOptions& rowReaderOptions() const {
    return options_;
  }

  std::shared_ptr<const dwio::common::TypeWithId> selectedType() const {
    if (!selectedSchema_) {
      selectedSchema_ = columnSelector_->buildSelected();
    }

    return selectedSchema_;
  }

  uint64_t rowNumber();

  uint64_t seekToRow(uint64_t rowNumber);

  uint64_t skipRows(uint64_t numberOfRowsToSkip);

  uint32_t currentStripe() const {
    return currentStripe_;
  }

  uint64_t getStrideIndex() const override {
    return strideIndex_;
  }

  /// Estimates the space used by the reader
  size_t estimatedReaderMemory() const;

  /// Estimates the row size for projected columns
  std::optional<size_t> estimatedRowSize() const override;

  /// Returns number of rows read. Guaranteed to be less then or equal to size.
  uint64_t next(
      uint64_t size,
      VectorPtr& result,
      const dwio::common::Mutation* = nullptr) override;

  void updateRuntimeStats(
      dwio::common::RuntimeStatistics& stats) const override {
    stats.skippedStrides += skippedStrides_;
    stats.processedStrides += processedStrides_;
    stats.footerBufferOverread += getReader().footerBufferOverread();
    stats.numStripes += stripeCeiling_ - firstStripe_;
    stats.columnReaderStatistics.flattenStringDictionaryValues +=
        columnReaderStatistics_.flattenStringDictionaryValues;
  }

  void resetFilterCaches() override;

  bool allPrefetchIssued() const override {
    return true;
  }

  // Returns the skipped strides for 'stripe'. Used for testing.
  std::optional<std::vector<uint64_t>> stridesToSkip(uint32_t stripe) const {
    auto it = stripeStridesToSkip_.find(stripe);
    if (it == stripeStridesToSkip_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void loadCurrentStripe();

  std::optional<std::vector<PrefetchUnit>> prefetchUnits() override {
    return std::nullopt;
  }

  int64_t nextRowNumber() override;

  int64_t nextReadSize(uint64_t size) override;

  std::shared_ptr<const RowType> type() const {
    if (columnSelector_) {
      return columnSelector_->getSchema();
    }
    return options_.requestedType();
  }

 private:
  bool shouldReadNode(uint32_t nodeId) const;

  std::optional<size_t> estimatedRowSizeHelper(
      const FooterWrapper& fileFooter,
      const dwio::common::Statistics& stats,
      uint32_t nodeId) const;

  bool emptyFile() const {
    return stripeCeiling_ == firstStripe_;
  }

  void checkSkipStrides(uint64_t strideSize);

  void readNext(
      uint64_t rowsToRead,
      const dwio::common::Mutation*,
      VectorPtr& result);

  uint64_t skip(uint64_t numValues);

  std::unique_ptr<ColumnReader>& getColumnReader();

  std::unique_ptr<dwio::common::SelectiveColumnReader>&
  getSelectiveColumnReader();

  std::unique_ptr<dwio::common::UnitLoader> getUnitLoader();

  const dwio::common::RowReaderOptions options_;
  dwio::common::ColumnReaderOptions columnReaderOptions_;

  // column selector
  const std::shared_ptr<dwio::common::ColumnSelector> columnSelector_;
  const std::function<void(std::chrono::high_resolution_clock::duration)>
      decodingTimeCallback_;

  // footer
  std::vector<uint64_t> firstRowOfStripe_;
  mutable std::shared_ptr<const dwio::common::TypeWithId> selectedSchema_;

  // reading state
  uint64_t previousRow_;
  uint32_t firstStripe_;
  uint32_t currentStripe_;
  // The the stripe AFTER the last one that should be read. e.g. if the highest
  // stripe in the RowReader's bounds is 3, then stripeCeiling_ is 4.
  uint32_t stripeCeiling_;
  uint64_t currentRowInStripe_;
  uint64_t rowsInCurrentStripe_;
  uint64_t strideIndex_;

  std::shared_ptr<BitSet> projectedNodes_;

  const uint64_t* stridesToSkip_;
  int stridesToSkipSize_;
  // Record of strides to skip in each visited stripe. Used for diagnostics.
  std::unordered_map<uint32_t, std::vector<uint64_t>> stripeStridesToSkip_;
  // Number of skipped strides.
  int64_t skippedStrides_{0};

  // Number of processed strides.
  int64_t processedStrides_{0};

  // Set to true after clearing filter caches, i.e. adding a dynamic filter.
  // Causes filters to be re-evaluated against stride stats on next stride
  // instead of next stripe.
  bool recomputeStridesToSkip_{false};

  dwio::common::ColumnReaderStatistics columnReaderStatistics_;

  std::optional<int64_t> nextRowNumber_;

  std::unique_ptr<dwio::common::UnitLoader> unitLoader_;
  DwrfUnit* currentUnit_;
};

class DwrfReader : public dwio::common::Reader {
 public:
  /**
   * Constructor that lets the user specify reader options and input stream.
   */
  DwrfReader(
      const dwio::common::ReaderOptions& options,
      std::unique_ptr<dwio::common::BufferedInput> input);

  ~DwrfReader() override = default;

  common::CompressionKind getCompression() const {
    return readerBase_->compressionKind();
  }

  WriterVersion getWriterVersion() const {
    return readerBase_->writerVersion();
  }

  const std::string& getWriterName() const {
    return readerBase_->writerName();
  }

  std::vector<std::string> getMetadataKeys() const;

  std::string getMetadataValue(const std::string& key) const;

  bool hasMetadataValue(const std::string& key) const;

  uint64_t getCompressionBlockSize() const {
    return readerBase_->compressionBlockSize();
  }

  uint32_t getNumberOfStripes() const {
    return readerBase_->footer().stripesSize();
  }

  std::vector<uint64_t> getRowsPerStripe() const {
    return readerBase_->rowsPerStripe();
  }
  uint32_t strideSize() const {
    return readerBase_->footer().rowIndexStride();
  }

  std::unique_ptr<StripeInformation> getStripe(uint32_t) const;

  uint64_t getFileLength() const {
    return readerBase_->fileLength();
  }

  std::unique_ptr<dwio::common::Statistics> getStatistics() const {
    return readerBase_->statistics();
  }

  std::unique_ptr<dwio::common::ColumnStatistics> columnStatistics(
      uint32_t nodeId) const override {
    return readerBase_->columnStatistics(nodeId);
  }

  const std::shared_ptr<const RowType>& rowType() const override {
    return readerBase_->schema();
  }

  const std::shared_ptr<const dwio::common::TypeWithId>& typeWithId()
      const override {
    return readerBase_->schemaWithId();
  }

  const PostScript& getPostscript() const {
    return readerBase_->postScript();
  }

  const FooterWrapper& getFooter() const {
    return readerBase_->footer();
  }

  std::optional<uint64_t> numberOfRows() const override {
    auto& fileFooter = readerBase_->footer();
    if (fileFooter.hasNumberOfRows()) {
      return fileFooter.numberOfRows();
    }
    return std::nullopt;
  }

  static uint64_t getMemoryUse(
      ReaderBase& readerBase,
      int32_t stripeIx,
      const dwio::common::ColumnSelector& cs);

  uint64_t getMemoryUse(int32_t stripeIx = -1);

  uint64_t getMemoryUseByFieldId(
      const std::vector<uint64_t>& include,
      int32_t stripeIx = -1);

  uint64_t getMemoryUseByName(
      const std::vector<std::string>& names,
      int32_t stripeIx = -1);

  uint64_t getMemoryUseByTypeId(
      const std::vector<uint64_t>& include,
      int32_t stripeIx = -1);

  std::unique_ptr<dwio::common::RowReader> createRowReader(
      const dwio::common::RowReaderOptions& options = {}) const override;

  std::unique_ptr<DwrfRowReader> createDwrfRowReader(
      const dwio::common::RowReaderOptions& options = {}) const;

  /**
   * Create a reader to the for the dwrf file.
   * @param input the stream to read
   * @param options the options for reading the file
   */
  static std::unique_ptr<DwrfReader> create(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options);

  ReaderBase* testingReaderBase() const {
    return readerBase_.get();
  }

 private:
  // Ensures that files column names match the ones from the table schema using
  // column indices.
  void updateColumnNamesFromTableSchema();

 private:
  std::shared_ptr<ReaderBase> readerBase_;
};

class DwrfReaderFactory : public dwio::common::ReaderFactory {
 public:
  DwrfReaderFactory() : ReaderFactory(dwio::common::FileFormat::DWRF) {}

  std::unique_ptr<dwio::common::Reader> createReader(
      std::unique_ptr<dwio::common::BufferedInput> input,
      const dwio::common::ReaderOptions& options) override {
    return DwrfReader::create(std::move(input), options);
  }
};

} // namespace facebook::velox::dwrf
