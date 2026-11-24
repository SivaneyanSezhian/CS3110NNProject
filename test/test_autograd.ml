open CS3110NNProject
module A = Autograd
module T = Tensor

let scalar = T.tensor_of
let vector = T.tensor_of_1d_list
let matrix = T.tensor_of_2d_list

let assert_close actual expected =
  if not (abs_float (actual -. expected) <= 1e-6 *. max 1. (abs_float expected))
  then failwith (Printf.sprintf "Expected %.12g, got %.12g" expected actual)

let assert_tensor actual expected =
  assert (T.get_dims actual = T.get_dims expected);
  ignore (T.pointwise2 assert_close actual expected)

let assert_assertion f =
  let raised =
    try
      ignore (f ());
      false
    with Assert_failure _ -> true
  in
  assert raised

let test_make_and_scalar_backward () =
  let value = matrix [ [ 1.; 2. ]; [ 3.; 4. ] ] in
  let node = A.make value in
  assert_tensor (A.value node) value;
  assert_tensor (A.grad node) (T.zeroes [ 2; 2 ]);
  let leaf = A.make (scalar 7.) in
  assert_tensor (A.grad leaf) (scalar 0.);
  A.backward leaf;
  assert_tensor (A.grad leaf) (scalar 1.)

let test_add_and_mse () =
  let a = A.make (matrix [ [ 1.; 2. ]; [ 3.; 4. ] ]) in
  let b = A.make (matrix [ [ 4.; 3. ]; [ 2.; 1. ] ]) in
  let sum = A.add a b in
  assert_tensor (A.value sum) (matrix [ [ 5.; 5. ]; [ 5.; 5. ] ]);
  let loss = A.mse_loss sum (matrix [ [ 1.; 3. ]; [ 5.; 7. ] ]) in
  assert_tensor (A.value loss) (scalar 6.);
  A.backward loss;
  let expected = matrix [ [ 2.; 1. ]; [ 0.; -1. ] ] in
  assert_tensor (A.grad sum) expected;
  assert_tensor (A.grad a) expected;
  assert_tensor (A.grad b) expected;
  assert_tensor (A.grad loss) (scalar 1.)

let test_sigmoid () =
  let x = A.make (vector [ -.log 3.; 0.; log 3. ]) in
  let output = A.sigmoid x in
  assert_tensor (A.value output) (vector [ 0.25; 0.5; 0.75 ]);
  A.backward (A.mse_loss output (T.zeroes [ 3 ]));
  assert_tensor (A.grad x) (vector [ 1. /. 32.; 1. /. 12.; 3. /. 32. ])

let test_relu () =
  let x = A.make (vector [ -2.; 0.; 3. ]) in
  let output = A.relu x in
  assert_tensor (A.value output) (vector [ 0.; 0.; 3. ]);
  (* A nonzero loss derivative at zero checks ReLU's chosen derivative. *)
  A.backward (A.mse_loss output (T.ones [ 3 ]));
  assert_tensor (A.grad x) (vector [ 0.; 0.; 4. /. 3. ])

let test_tanh () =
  let x = A.make (vector [ -1.; 0.; 1. ]) in
  let output = A.tanh x in
  let y = 0.7615941559557649 in
  assert_tensor (A.value output) (vector [ -.y; 0.; y ]);
  A.backward (A.mse_loss output (T.zeroes [ 3 ]));
  let derivative = 2. /. 3. *. y *. (1. -. (y *. y)) in
  assert_tensor (A.grad x) (vector [ -.derivative; 0.; derivative ]);
  let zero = A.make (scalar 0.) in
  A.backward (A.tanh zero);
  assert_tensor (A.grad zero) (scalar 1.);
  let large = A.make (vector [ -1000.; 1000. ]) in
  let saturated = A.tanh large in
  assert_tensor (A.value saturated) (vector [ -1.; 1. ]);
  A.backward (A.mse_loss saturated (T.zeroes [ 2 ]));
  assert_tensor (A.grad large) (T.zeroes [ 2 ])

let test_rectangular_mlp () =
  let w = A.make (matrix [ [ 1.; 2.; 0. ]; [ 3.; 4.; -1. ] ]) in
  let x = A.make (matrix [ [ 2. ]; [ 1. ]; [ 3. ] ]) in
  let b = A.make (matrix [ [ 1. ]; [ 1. ] ]) in
  let product = A.matmul w x in
  assert_tensor (A.value product) (matrix [ [ 4. ]; [ 7. ] ]);
  let prediction = A.relu (A.add product b) in
  assert_tensor (A.value prediction) (matrix [ [ 5. ]; [ 8. ] ]);
  let loss = A.mse_loss prediction (matrix [ [ 2. ]; [ 3. ] ]) in
  assert_tensor (A.value loss) (scalar 17.);
  A.backward loss;
  assert_tensor (A.grad w) (matrix [ [ 6.; 3.; 9. ]; [ 10.; 5.; 15. ] ]);
  assert_tensor (A.grad x) (matrix [ [ 18. ]; [ 26. ]; [ -5. ] ]);
  assert_tensor (A.grad b) (matrix [ [ 3. ]; [ 5. ] ])

let test_shared_graph () =
  let x = A.make (scalar 2.) in
  let shared = A.add x x in
  let left = A.relu shared in
  let right = A.add shared (A.make (scalar 1.)) in
  let loss = A.mse_loss (A.add left right) (scalar 0.) in
  A.backward loss;
  assert_tensor (A.value loss) (scalar 81.);
  assert_tensor (A.grad shared) (scalar 36.);
  assert_tensor (A.grad x) (scalar 72.)

let test_distinct_equal_nodes () =
  let a = A.make (scalar 3.) in
  let b = A.make (scalar 3.) in
  A.backward (A.mse_loss (A.add a b) (scalar 0.));
  assert_tensor (A.grad a) (scalar 12.);
  assert_tensor (A.grad b) (scalar 12.)

let test_repeated_backward () =
  let x = A.make (scalar 3.) in
  let doubled = A.add x x in
  let loss = A.mse_loss doubled (scalar 0.) in
  A.backward loss;
  assert_tensor (A.grad x) (scalar 24.);
  A.backward loss;
  assert_tensor (A.grad doubled) (scalar 12.);
  assert_tensor (A.grad x) (scalar 24.);
  A.backward (A.mse_loss x (scalar 1.));
  assert_tensor (A.grad x) (scalar 4.);
  (* The second loss does not reach doubled, so its stored gradient survives. *)
  assert_tensor (A.grad doubled) (scalar 12.)

let test_nested_mse () =
  let x = A.make (scalar 2.) in
  let inner = A.mse_loss x (scalar 0.) in
  let outer = A.mse_loss inner (scalar 0.) in
  A.backward outer;
  assert_tensor (A.value outer) (scalar 16.);
  assert_tensor (A.grad inner) (scalar 8.);
  assert_tensor (A.grad x) (scalar 32.)

let test_bce_loss () =
  let pred = A.make (vector [ 0.8; 0.2 ]) in
  let target = T.tensor_of_1d_list [ 1.; 0. ] in
  let loss = A.bce_loss pred target in
  assert_tensor (A.value loss) (scalar (-.log 0.8));
  A.backward loss;
  assert_tensor (A.grad pred) (vector [ -0.625; 0.625 ]);
  A.backward (A.add loss loss);
  assert_tensor (A.grad pred) (vector [ -1.25; 1.25 ]);
  (* Check that the backward rule differentiates the actual forward value. *)
  let loss_at p =
    T.lookup (A.value (A.bce_loss (A.make (scalar p)) (scalar 0.3))) []
  in
  let p = A.make (scalar 0.6) in
  A.backward (A.bce_loss p (scalar 0.3));
  let h = 1e-6 in
  assert_close
    (T.lookup (A.grad p) [])
    ((loss_at (0.6 +. h) -. loss_at (0.6 -. h)) /. (2. *. h))

let test_bce_endpoints () =
  let assert_finite x =
    assert (classify_float x <> FP_nan && classify_float x <> FP_infinite)
  in
  List.iter
    (fun (p, y) ->
      let pred = A.make (scalar p) in
      let loss = A.bce_loss pred (scalar y) in
      assert_finite (T.lookup (A.value loss) []);
      A.backward loss;
      assert_tensor (A.grad pred) (scalar 0.))
    [ (0., 0.); (0., 1.); (1., 0.); (1., 1.) ];
  let logits = A.make (vector [ -1000.; 1000. ]) in
  let loss = A.bce_loss (A.sigmoid logits) (vector [ 1.; 0. ]) in
  assert_finite (T.lookup (A.value loss) []);
  A.backward loss;
  assert_tensor (A.grad logits) (T.zeroes [ 2 ])

let test_invalid_shapes () =
  let a = A.make (T.ones [ 2; 3 ]) in
  let b = A.make (T.ones [ 3; 2 ]) in
  let v = A.make (T.ones [ 3 ]) in
  let s = A.make (scalar 1.) in
  assert_assertion (fun () -> A.add a b);
  assert_assertion (fun () -> A.matmul a a);
  assert_assertion (fun () -> A.matmul v b);
  assert_assertion (fun () -> A.matmul a v);
  assert_assertion (fun () -> A.matmul s b);
  assert_assertion (fun () -> A.matmul a s);
  assert_assertion (fun () -> A.mse_loss a (T.ones [ 3; 2 ]));
  assert_assertion (fun () -> A.backward a)

let () =
  List.iter
    (fun (name, test) ->
      try test ()
      with exn ->
        failwith (Printf.sprintf "%s: %s" name (Printexc.to_string exn)))
    [
      ("make and scalar backward", test_make_and_scalar_backward);
      ("addition and MSE", test_add_and_mse);
      ("sigmoid", test_sigmoid);
      ("ReLU", test_relu);
      ("tanh", test_tanh);
      ("rectangular MLP", test_rectangular_mlp);
      ("shared graph", test_shared_graph);
      ("distinct equal nodes", test_distinct_equal_nodes);
      ("repeated backward", test_repeated_backward);
      ("nested MSE", test_nested_mse);
      ("BCE loss and gradients", test_bce_loss);
      ("BCE endpoints", test_bce_endpoints);
      ("invalid shapes", test_invalid_shapes);
    ];
  print_endline "Autograd module tests passed"
