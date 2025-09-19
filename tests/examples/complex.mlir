// Generated MLIR code from Python source
func.func @fibonacci(%arg0: i32) -> i32 {
  %0 = arith.constant 0 : i32
  %1 = arith.cmpi sle, %arg0, %0 : i32
  cf.cond_br %1, ^then0, ^else1
then0:
  %2 = arith.constant 0 : i32
  func.return %2 : i32
  cf.br ^if_end2
else1:
  %3 = arith.constant 1 : i32
  %4 = arith.cmpi eq, %arg0, %3 : i32
  cf.cond_br %4, ^then3, ^else4
then3:
  %5 = arith.constant 1 : i32
  func.return %5 : i32
  cf.br ^if_end5
else4:
  %6 = arith.constant 0 : i32
  %7 = arith.constant 1 : i32
  %8 = arith.constant 2 : i32
  cf.br ^loop_header6
loop_header6:
  %9 = arith.cmpi sle, %8, %arg0 : i32
  cf.cond_br %9, ^loop_body7, ^loop_end8
loop_body7:
  %10 = arith.addi %6, %7 : i32
  %11 = arith.constant 1 : i32
  %12 = arith.addi %8, %11 : i32
  cf.br ^loop_header6
loop_end8:
  func.return %10 : i32
  cf.br ^if_end5
if_end5:
  cf.br ^if_end2
if_end2:
}
func.func @is_even(%arg0: i32) -> i32 {
  %13 = arith.constant 2 : i32
  %14 = arith.remsi %arg0, %13 : i32
  %15 = arith.constant 0 : i32
  %16 = arith.cmpi eq, %14, %15 : i32
  func.return %16 : i32
}
func.func @calculate_sum(%arg0: i32) -> i32 {
  %17 = arith.constant 0 : i32
  %18 = arith.constant 1 : i32
  cf.br ^loop_header9
loop_header9:
  %19 = arith.cmpi sle, %18, %arg0 : i32
  cf.cond_br %19, ^loop_body10, ^loop_end11
loop_body10:
  %20 = func.call @is_even(%18) : (i32) -> i32
  cf.cond_br %20, ^then12, ^if_end13
then12:
  %21 = arith.addi %17, %18 : i32
  cf.br ^if_end13
if_end13:
  %22 = arith.constant 1 : i32
  %23 = arith.addi %18, %22 : i32
  cf.br ^loop_header9
loop_end11:
  func.return %21 : i32
}