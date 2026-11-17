open CS3110NNProject
module A = Autograd
module M = Mlp
module T = Tensor

let matrix = T.tensor_of_2d_list

let assert_invalid f =
  let raised =
    try
      ignore (f ());
      false
    with Invalid_argument _ -> true
  in
  assert raised

let net =
  M.create ~input_dim:2
    [
      { M.units = 3; activation = M.Relu };
      { M.units = 1; activation = M.Identity };
    ]

let test_create () =
  assert (M.input_dim net = 2);
  assert (M.output_dim net = 1);
  (* 2 * 3 + 3 in the first layer and 3 * 1 + 1 in the second. *)
  assert (M.num_parameters net = 13)

let test_invalid_create () =
  let relu = { M.units = 1; activation = M.Relu } in
  assert_invalid (fun () -> M.create ~input_dim:0 [ relu ]);
  assert_invalid (fun () -> M.create ~input_dim:2 []);
  assert_invalid (fun () ->
      M.create ~input_dim:2 [ { M.units = 0; activation = M.Relu } ])

let test_init_parameters () =
  let params = M.init_parameters ~seed:0 net in
  assert (
    List.map T.get_dims params = [ [ 2; 3 ]; [ 1; 3 ]; [ 3; 1 ]; [ 1; 1 ] ]);
  assert (List.nth params 1 = T.zeroes [ 1; 3 ]);
  assert (List.nth params 3 = T.zeroes [ 1; 1 ]);
  assert (M.init_parameters ~seed:0 net = params)

let test_forward () =
  let small =
    M.create ~input_dim:2
      [
        { M.units = 2; activation = M.Relu };
        { M.units = 1; activation = M.Identity };
      ]
  in
  let params =
    [
      matrix [ [ 1.; -1. ]; [ 2.; 1. ] ];
      matrix [ [ 0.5; -0.5 ] ];
      matrix [ [ 1. ]; [ 2. ] ];
      matrix [ [ -1. ] ];
    ]
  in
  let inputs = matrix [ [ 1.; 2. ]; [ 3.; -1. ] ] in
  let output = M.forward small (List.map A.make params) inputs in
  (* The hidden rows are [5.5; 0.5] and [1.5; -4.5] before ReLU, so the second
     row loses its negative unit. *)
  assert (A.value output = matrix [ [ 5.5 ]; [ 0.5 ] ])

let test_activations () =
  let run activation =
    let single = M.create ~input_dim:1 [ { M.units = 1; activation } ] in
    let params = [ A.make (matrix [ [ 1. ] ]); A.make (matrix [ [ 0. ] ]) ] in
    T.lookup (A.value (M.forward single params (matrix [ [ -2. ] ]))) [ 0; 0 ]
  in
  assert (run M.Identity = -2.);
  assert (run M.Relu = 0.);
  assert (run M.Sigmoid = 1. /. (1. +. exp 2.));
  assert (run M.Tanh = tanh (-2.))

let test_invalid_forward () =
  let params = List.map A.make (M.init_parameters ~seed:0 net) in
  let inputs = matrix [ [ 1.; 2. ] ] in
  assert_invalid (fun () -> M.forward net params (matrix [ [ 1. ] ]));
  assert_invalid (fun () -> M.forward net (List.tl params) inputs);
  assert_invalid (fun () -> M.forward net (List.rev params) inputs)

let test_strings () =
  assert (M.activation_of_string " ReLU " = Some M.Relu);
  assert (M.activation_of_string "linear" = Some M.Identity);
  assert (M.activation_of_string "softmax" = None);
  assert (M.string_of_activation M.Tanh = "tanh");
  assert (
    M.to_string net
    = "input: 2\n\
       layer 1: 2 -> 3, relu\n\
       layer 2: 3 -> 1, identity\n\
       parameters: 13")

let () =
  List.iter
    (fun (name, test) ->
      try test ()
      with exn ->
        failwith (Printf.sprintf "%s: %s" name (Printexc.to_string exn)))
    [
      ("create", test_create);
      ("invalid create", test_invalid_create);
      ("init parameters", test_init_parameters);
      ("forward", test_forward);
      ("activations", test_activations);
      ("invalid forward", test_invalid_forward);
      ("strings", test_strings);
    ];
  print_endline "Mlp module tests passed"
