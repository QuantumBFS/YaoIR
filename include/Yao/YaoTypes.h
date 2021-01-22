#pragma once
#include <assert.h>
#include <cstddef>
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"

namespace mlir {
  namespace yao {
    
class GateTypeStorage : public ::mlir::TypeStorage {
  GateTypeStorage(size_t numQubits)
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
  static GateTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<GateTypeStorage>())
        GateTypeStorage(key);
  }
public:
  /// The parametric data held by the storage class.
  size_t numQubits;
};


class GateType : public ::mlir::Type::TypeBase<GateType, mlir::Type,
                                          GateTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This method is used to get an instance of the 'GateType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static GateType get(mlir::MLIRContext *ctx, size_t numQubits) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // context.
    return Base::get(ctx, numQubits);
  }

  /// This method is used to get an instance of the 'GateType', defined at
  /// the given location. If any of the construction invariants are invalid,
  /// errors are emitted with the provided location and a null type is returned.
  /// Note: This method is completely optional.
  static GateType getChecked(mlir::Location location, size_t numQubits) {
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