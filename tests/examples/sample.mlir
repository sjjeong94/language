// Generated MLIR code from Python source
func.func @add(%arg0: i32, %arg1: i32) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  func.return %0 : i32
}
func.func @factorial(%arg0: i32) -> i32 {
  %1 = arith.constant 1 : i32
  %2 = arith.cmpi sle, %arg0, %1 : i32
  cf.cond_br %2, ^then0, ^else1
then0:
  %3 = arith.constant 1 : i32
  func.return %3 : i32
  cf.br ^if_end2
else1:
  %4 = arith.constant 1 : i32
  %5 = arith.subi %arg0, %4 : i32
  %6 = func.call @factorial(%5) : (i32) -> i32
  %7 = arith.muli %arg0, %6 : i32
  func.return %7 : i32
  cf.br ^if_end2
if_end2:
}
func.func @main() -> i32 {
  %8 = arith.constant 5 : i32
  %9 = arith.constant 3 : i32
  %10 = func.call @add(%8, %9) : (i32, i32) -> i32
  %11 = arith.constant 4 : i32
  %12 = func.call @factorial(%11) : (i32) -> i32
  %13 = arith.addi %10, %12 : i32
  func.return %13 : i32
}