#pragma once
#include <assert.h>
#include <cstddef>
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

namespace mlir {
  namespace yao {
    
class OperatorTypeStorage : public ::mlir::TypeStorage {
  OperatorTypeStorage(size_t numQubits)
      : numQubits(numQubits) {}
public:
  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = size_t;
  
  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == numQubits;
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_code(key);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(size_t numQubits) {
    return numQubits;
  }

  /// Define a construction method for creating a new instance of this storage.
  static OperatorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<OperatorTypeStorage>())
        OperatorTypeStorage(key);
  }
public:
  /// The parametric data held by the storage class.
  size_t numQubits;
};


class OperatorType : public ::mlir::Type::TypeBase<OperatorType, mlir::Type,
                                          OperatorTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'OperatorType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static OperatorType get(mlir::MLIRContext *ctx, size_t numQubits) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(ctx, numQubits);
  }

  /// This method is used to get an instance of the 'OperatorType', defined at
  /// the given location. If any of the construction invariants are invalid,
  /// errors are emitted with the provided location and a null type is returned.
  /// Note: This method is completely optional.
  static OperatorType getChecked(mlir::Location location, size_t numQubits) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the location.
    return Base::getChecked(location, numQubits);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      const mlir::AttributeStorage*, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }


  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }

  /// Return the parameter value.
  size_t getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->numQubits;
  }
};

class CtrlFlagTypeStorage : public ::mlir::TypeStorage {
  CtrlFlagTypeStorage(size_t numQubits)
      : numQubits(numQubits) {}
public:
  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = size_t;
  
  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == numQubits;
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_code(key);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(size_t numQubits) {
    return numQubits;
  }

  /// Define a construction method for creating a new instance of this storage.
  static CtrlFlagTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<CtrlFlagTypeStorage>())
        CtrlFlagTypeStorage(key);
  }
public:
  /// The parametric data held by the storage class.
  size_t numQubits;
};

class CtrlFlagType : public ::mlir::Type::TypeBase<CtrlFlagType, mlir::Type,
                                          CtrlFlagTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'CtrlFlagType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static CtrlFlagType get(mlir::MLIRContext *ctx, size_t numQubits) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(ctx, numQubits);
  }

  /// This method is used to get an instance of the 'CtrlFlagType', defined at
  /// the given location. If any of the construction invariants are invalid,
  /// errors are emitted with the provided location and a null type is returned.
  /// Note: This method is completely optional.
  static CtrlFlagType getChecked(mlir::Location location, size_t numQubits) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the location.
    return Base::getChecked(location, numQubits);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      const mlir::AttributeStorage*, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }


  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }

  /// Return the parameter value.
  size_t getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->numQubits;
  }
};


class LocationsTypeStorage : public ::mlir::TypeStorage {
  LocationsTypeStorage(size_t numQubits)
      : numQubits(numQubits) {}
public:
  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = size_t;
  
  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == numQubits;
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_code(key);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(size_t numQubits) {
    return numQubits;
  }

  /// Define a construction method for creating a new instance of this storage.
  static LocationsTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<LocationsTypeStorage>())
        LocationsTypeStorage(key);
  }
public:
  /// The parametric data held by the storage class.
  size_t numQubits;
};


class LocationsType : public ::mlir::Type::TypeBase<LocationsType, mlir::Type,
                                          LocationsTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'LocationsType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static LocationsType get(mlir::MLIRContext *ctx, size_t numQubits) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(ctx, numQubits);
  }

  /// This method is used to get an instance of the 'LocationsType', defined at
  /// the given location. If any of the construction invariants are invalid,
  /// errors are emitted with the provided location and a null type is returned.
  /// Note: This method is completely optional.
  static LocationsType getChecked(mlir::Location location, size_t numQubits) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the location.
    return Base::getChecked(location, numQubits);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      const mlir::AttributeStorage*, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }


  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }

  /// Return the parameter value.
  size_t getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->numQubits;
  }
};

class MeasureResultTypeStorage : public ::mlir::TypeStorage {
  MeasureResultTypeStorage(size_t numQubits)
      : numQubits(numQubits) {}
public:
  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = size_t;
  
  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == numQubits;
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_code(key);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(size_t numQubits) {
    return numQubits;
  }

  /// Define a construction method for creating a new instance of this storage.
  static MeasureResultTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<MeasureResultTypeStorage>())
        MeasureResultTypeStorage(key);
  }
public:
  /// The parametric data held by the storage class.
  size_t numQubits;
};


class MeasureResultType : public ::mlir::Type::TypeBase<MeasureResultType, mlir::Type,
                                          MeasureResultTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'MeasureResultType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static MeasureResultType get(mlir::MLIRContext *ctx, size_t numQubits) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(ctx, numQubits);
  }

  /// This method is used to get an instance of the 'MeasureResultType', defined at
  /// the given location. If any of the construction invariants are invalid,
  /// errors are emitted with the provided location and a null type is returned.
  /// Note: This method is completely optional.
  static MeasureResultType getChecked(mlir::Location location, size_t numQubits) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the location.
    return Base::getChecked(location, numQubits);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      const mlir::AttributeStorage*, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }


  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, size_t numQubits) {
    // Our type only allows non-zero parameters.
    if (numQubits == 0)
      return mlir::failure();
    return mlir::success();
  }

  /// Return the parameter value.
  size_t getNumQubits() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->numQubits;
  }
};


  }
}