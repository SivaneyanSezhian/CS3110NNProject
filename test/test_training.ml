open CS3110NNProject

let close a b = abs_float (a -. b) < 0.000001
let assert_close actual expected = assert (close actual expected)

let single_layer_forward nodes inputs =
  match nodes with
  | [ w ] -> Autograd.matmul (Autograd.make inputs) w
  | _ -> failwith "Expected one weight matrix"

let two_layer_forward nodes inputs =
  match nodes with
  | [ w1; w2 ] ->
      let hidden = Autograd.relu (Autograd.matmul (Autograd.make inputs) w1) in
      Autograd.matmul hidden w2
  | _ -> failwith "Expected two weight matrices"

let test_single_step () =
  let weights = Tensor.tensor_of_2d_list [ [ 1.0 ] ] in
  let inputs = Tensor.tensor_of_2d_list [ [ 2.0 ] ] in
  let targets = Tensor.tensor_of_2d_list [ [ 4.0 ] ] in
  let updated, before =
    Training.train_step ~learning_rate:0.1 ~forward:single_layer_forward
      ~parameters:[ weights ] ~inputs ~targets
  in
  assert_close before 4.0;
  (match updated with
  | [ w ] -> assert_close (Tensor.lookup w [ 0; 0 ]) 1.8
  | _ -> assert false);
  assert_close (Tensor.lookup weights [ 0; 0 ]) 1.0;
  assert_close
    (Training.evaluate ~forward:single_layer_forward ~parameters:updated ~inputs
       ~targets)
    0.16

let test_training_loop_and_inference () =
  let weights = Tensor.tensor_of_2d_list [ [ 1.0 ] ] in
  let inputs = Tensor.tensor_of_2d_list [ [ 2.0 ] ] in
  let targets = Tensor.tensor_of_2d_list [ [ 4.0 ] ] in
  let updated, history =
    Training.train ~learning_rate:0.1 ~epochs:3 ~forward:single_layer_forward
      ~parameters:[ weights ] ~inputs ~targets ~validation:(inputs, targets) ()
  in
  (match updated with
  | [ w ] -> assert_close (Tensor.lookup w [ 0; 0 ]) 1.992
  | _ -> assert false);
  assert (List.length history.train_losses = 3);
  List.iter2 assert_close history.train_losses [ 0.16; 0.0064; 0.000256 ];
  (match history.validation_losses with
  | Some losses -> List.iter2 assert_close losses history.train_losses
  | None -> assert false);
  let pred =
    Training.predict ~forward:single_layer_forward ~parameters:updated ~inputs
  in
  assert_close (Tensor.lookup pred [ 0; 0 ]) 3.984

let test_two_weight_matrices () =
  let w1 = Tensor.tensor_of_2d_list [ [ 1.0; 0.5 ] ] in
  let w2 = Tensor.tensor_of_2d_list [ [ 1.0 ]; [ 1.0 ] ] in
  let inputs = Tensor.tensor_of_2d_list [ [ 2.0 ] ] in
  let targets = Tensor.tensor_of_2d_list [ [ 4.0 ] ] in
  let before =
    Training.evaluate ~forward:two_layer_forward ~parameters:[ w1; w2 ] ~inputs
      ~targets
  in
  assert_close before 1.0;
  let updated, step_loss =
    Training.train_step ~learning_rate:0.01 ~forward:two_layer_forward
      ~parameters:[ w1; w2 ] ~inputs ~targets
  in
  assert_close step_loss before;
  assert (List.length updated = 2);
  let after =
    Training.evaluate ~forward:two_layer_forward ~parameters:updated ~inputs
      ~targets
  in
  assert (after < before)

let test_invalid_hyperparameters () =
  let weights = Tensor.tensor_of_2d_list [ [ 1.0 ] ] in
  let inputs = Tensor.tensor_of_2d_list [ [ 2.0 ] ] in
  let targets = Tensor.tensor_of_2d_list [ [ 4.0 ] ] in
  let bad_learning_rate =
    try
      ignore
        (Training.train_step ~learning_rate:0.0 ~forward:single_layer_forward
           ~parameters:[ weights ] ~inputs ~targets);
      false
    with Invalid_argument _ -> true
  in
  assert bad_learning_rate;
  let bad_epochs =
    try
      ignore
        (Training.train ~learning_rate:0.1 ~epochs:0
           ~forward:single_layer_forward ~parameters:[ weights ] ~inputs
           ~targets ());
      false
    with Invalid_argument _ -> true
  in
  assert bad_epochs

let () =
  test_single_step ();
  test_training_loop_and_inference ();
  test_two_weight_matrices ();
  test_invalid_hyperparameters ();
  print_endline "Training module tests passed"
