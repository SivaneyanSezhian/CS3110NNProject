module Tensor = CS3110NNProject.Tensor

let tensor_equal tnsr1 tnsr2 =
  Tensor.check_dims tnsr1;
  Tensor.check_dims tnsr2;
  tnsr1 = tnsr2

let print_tensor_equal tnsr1 tnsr2 =
  Tensor.check_dims tnsr1;
  Tensor.check_dims tnsr2;
  if tnsr1 = tnsr2 then true
  else (
    print_endline (Tensor.string_of_tensor tnsr1);
    print_endline (Tensor.string_of_tensor tnsr2);
    false)

let assert_tensor_approx tnsr1 tnsr2 =
  ignore
    (Tensor.pointwise2
       (fun x y ->
         assert (
           abs_float (x -. y)
           <= max (abs_float (1e-3 *. x)) (abs_float (1e-3 *. y))))
       tnsr1 tnsr2)

let make_tensors () =
  assert (Tensor.zeroes [] = Tensor.tensor_of 0.);
  assert (
    Tensor.zeroes [ 2; 3 ]
    = Tensor.tensor_of_2d_list [ [ 0.; 0.; 0. ]; [ 0.; 0.; 0. ] ]);
  assert (
    Tensor.ones [ 3; 2 ]
    = Tensor.tensor_of_2d_array [| [| 1.; 1. |]; [| 1.; 1. |]; [| 1.; 1. |] |]);
  assert (Tensor.zeroes [ 5 ] = Tensor.tensor_of_1d_list [ 0.; 0.; 0.; 0.; 0. ]);
  assert (Tensor.ones [ 1 ] = Tensor.tensor_of_1d_array [| 1. |]);
  assert (
    Tensor.make_tensor [ 3; 3 ] (fun d ->
        match d with
        | x :: y :: _ -> x + (2 * y)
        | _ -> failwith "")
    = Tensor.tensor_of_2d_list [ [ 0; 2; 4 ]; [ 1; 3; 5 ]; [ 2; 4; 6 ] ])

let make_tensors_equal () =
  assert (
    tensor_equal
      (Tensor.zeroes [ 2; 3 ])
      (Tensor.tensor_of_2d_list [ [ 0.; 0.; 0. ]; [ 0.; 0.; 0. ] ]));
  assert (
    tensor_equal
      (Tensor.ones [ 3; 2 ])
      (Tensor.tensor_of_2d_array [| [| 1.; 1. |]; [| 1.; 1. |]; [| 1.; 1. |] |]));
  assert (
    tensor_equal (Tensor.zeroes [ 5 ])
      (Tensor.tensor_of_1d_list [ 0.; 0.; 0.; 0.; 0. ]));
  assert (tensor_equal (Tensor.ones [ 1 ]) (Tensor.tensor_of_1d_array [| 1. |]));
  assert (
    tensor_equal
      (Tensor.make_tensor [ 3; 3 ] (fun d ->
           match d with
           | x :: y :: _ -> x + (2 * y)
           | _ -> failwith ""))
      (Tensor.tensor_of_2d_list [ [ 0; 2; 4 ]; [ 1; 3; 5 ]; [ 2; 4; 6 ] ]))

let tensor_ops () =
  assert (
    tensor_equal
      (Tensor.pointwise
         (fun x -> 0. <= x && x <= 1.)
         (Tensor.random [ 5; 5; 5 ]))
      (Tensor.make_tensor [ 5; 5; 5 ] (fun _ -> true)));
  assert (
    tensor_equal
      (Tensor.pointwise2
         (fun x y -> 0. <= x +. y && x +. y <= 2.)
         (Tensor.random [ 5; 5; 5 ])
         (Tensor.random [ 5; 5; 5 ]))
      (Tensor.make_tensor [ 5; 5; 5 ] (fun _ -> true)));
  let tnsr = Tensor.random [ 5; 5; 5 ] in
  assert_tensor_approx
    (Tensor.pointwisen
       (fun xs -> List.fold_left (fun acc x -> acc +. x) 0. xs)
       (List.init 10 (fun _ -> tnsr)))
    (Tensor.pointwise (fun x -> x *. 10.) tnsr)

let tensor_ops_2 () =
  assert (
    tensor_equal
      (Tensor.flatten (Tensor.tensor_of_2d_list [ [ 1; 2 ]; [ 4; 5 ] ]))
      (Tensor.tensor_of_1d_list [ 1; 2; 4; 5 ]));
  assert (
    (Tensor.lookup (Tensor.tensor_of_2d_list [ [ 1; 2 ]; [ 4; 5 ] ])) [ 0; 1 ]
    = 2);
  assert (
    (Tensor.lookup (Tensor.tensor_of_2d_list [ [ 1; 2 ]; [ 4; 5 ] ])) [ 1; 1 ]
    = 5);
  assert (
    tensor_equal
      (Tensor.matmul
         (Tensor.tensor_of_2d_list [ [ 1.; 2. ]; [ 1.; 0. ] ])
         (Tensor.tensor_of_2d_list [ [ 1.; 0. ]; [ 0.; 2. ] ]))
      (Tensor.tensor_of_2d_list [ [ 1.; 4. ]; [ 1.; 0. ] ]));
  assert (
    tensor_equal
      (Tensor.stack_array
         [|
           Tensor.tensor_of_1d_list [ 1.; 0. ];
           Tensor.tensor_of_1d_list [ 0.; 2. ];
         |])
      (Tensor.tensor_of_2d_list [ [ 1.; 0. ]; [ 0.; 2. ] ]));
  assert (
    tensor_equal
      (Tensor.stack_list
         [
           Tensor.tensor_of_1d_list [ 1.; 0. ];
           Tensor.tensor_of_1d_list [ 0.; 2. ];
         ])
      (Tensor.tensor_of_2d_list [ [ 1.; 0. ]; [ 0.; 2. ] ]));
  assert (
    tensor_equal
      (Tensor.swap_dims
         (Tensor.tensor_of_2d_list [ [ 1.; 1. ]; [ 0.; 2. ] ])
         0 1)
      (Tensor.tensor_of_2d_list [ [ 1.; 0. ]; [ 1.; 2. ] ]));
  assert (
    tensor_equal
      (Tensor.permute_dims
         (Tensor.tensor_of_2d_list [ [ 1.; 1. ]; [ 0.; 2. ] ])
         [ 1; 0 ])
      (Tensor.tensor_of_2d_list [ [ 1.; 0. ]; [ 1.; 2. ] ]));
  assert (
    tensor_equal
      (Tensor.reshape
         (Tensor.tensor_of_2d_list [ [ 1.; 2. ]; [ 4.; 5. ] ])
         [ 4; 1 ])
      (Tensor.tensor_of_2d_list [ [ 1. ]; [ 2. ]; [ 4. ]; [ 5. ] ]))

let () =
  List.iter
    (fun (name, test) ->
      try test ()
      with exn ->
        failwith (Printf.sprintf "%s: %s" name (Printexc.to_string exn)))
    [
      ("make_tensors", make_tensors);
      ("make_tensors_equal", make_tensors_equal);
      ("tensor_ops", tensor_ops);
      ("tensor_ops_2", tensor_ops_2);
    ];
  print_endline "Tensor module tests passed"
