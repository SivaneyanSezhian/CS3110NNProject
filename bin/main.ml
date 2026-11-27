open CS3110NNProject
module T = Tensor
module M = Mlp
module Tr = Training

let learning_rate, epochs =
  let learning_rate = ref 0.01 in
  let epochs = ref 100 in
  Arg.parse
    [
      ( "--learning-rate",
        Arg.Float
          (fun rate ->
            if rate <= 0. || not (Float.is_finite rate) then
              raise (Arg.Bad "learning rate must be positive and finite");
            learning_rate := rate),
        "RATE Learning rate (default 0.01)" );
      ( "--epochs",
        Arg.Int
          (fun count ->
            if count <= 0 then raise (Arg.Bad "epochs must be positive");
            epochs := count),
        "COUNT Number of epochs (default 100)" );
    ]
    (fun argument -> raise (Arg.Bad ("Unexpected argument: " ^ argument)))
    "Usage: CS3110NNProject [--learning-rate RATE] [--epochs COUNT]";
  (!learning_rate, !epochs)

(** [data_of_csv file_path] converts the CSV file located at [file_path] to a 2d
    list of floats, ignoring any blank lines. *)
let data_of_csv file_path =
  file_path |> Csv.load
  |> List.filter (fun row -> row <> [])
  |> List.map (fun row -> List.map Float.of_string row)

(** [split lst pos] returns a tuple whose first element is a list containing the
    first [pos] elements in [lst] and whose second element is a list containing
    the remaining elements in [lst]. Raises [Failure] if [pos] is negative or
    exceeds the length of [lst]. *)
let split lst pos =
  let rec split_aux acc lst remaining =
    if remaining < 0 then failwith "pos must be a nonnegative integer"
    else if remaining = 0 then (List.rev acc, lst)
    else
      match lst with
      | [] -> failwith "not enough elements in list"
      | h :: t -> split_aux (h :: acc) t (remaining - 1)
  in
  split_aux [] lst pos

(** [pop lst] returns a tuple whose first element is a list including all but
    the last element and whose second element is a list including just the last
    element. Raises [Failure] if [lst] is empty. *)
let pop lst =
  try split lst (List.length lst - 1)
  with Failure _ -> failwith "cannot pop from empty list"

(** Represents each row in [data] as a tuple whose first element is the training
    instance in list form and whose second element is a list including just the
    label. *)
let data_split = "data/sample.csv" |> data_of_csv |> List.map pop

let x = data_split |> List.map fst
let y = data_split |> List.map snd
let x_train_val_list, x_test_list = split x 9000
let x_train_list, x_val_list = split x_train_val_list 8000
let x_train = T.tensor_of_2d_list x_train_list
let x_val = T.tensor_of_2d_list x_val_list
let x_test = T.tensor_of_2d_list x_test_list
let y_train_val_list, y_test_list = split y 9000
let y_train_list, y_val_list = split y_train_val_list 8000
let y_train = T.tensor_of_2d_list y_train_list
let y_val = T.tensor_of_2d_list y_val_list
let y_test = T.tensor_of_2d_list y_test_list

let layer_list =
  [
    { M.units = 256; M.activation = Relu };
    { M.units = 1; M.activation = Identity };
  ]

let net = M.create ~input_dim:10 layer_list
let params = M.init_parameters ~seed:42 net
let forward = M.forward net

(** Notifies the user of the selected optimization hyperparameters. *)
let training_message () =
  Printf.printf "Training for %d epochs with learning rate %g\n%!" epochs
    learning_rate

let trained_parameters, history =
  Tr.train ~loss_fn:Autograd.mse_loss ~validation:(x_val, y_val) ~learning_rate ~epochs ~forward
    ~parameters:params ~inputs:x_train ~targets:y_train ()

(** Reports the results of the trained model to the user. *)
let report () =
  Printf.printf "\nFinal epoch (%d)\nTraining MSE: %.6f\n" epochs
    (List.hd (List.rev history.train_losses));
  (match history.validation_losses with
  | None -> ()
  | Some losses ->
      Printf.printf "Validation MSE: %.6f\n" (List.hd (List.rev losses)));
  Printf.printf "Test MSE: %.6f\n"
    (Tr.evaluate ~loss_fn:Autograd.mse_loss ~forward ~parameters:trained_parameters ~inputs:x_test
       ~targets:y_test)
