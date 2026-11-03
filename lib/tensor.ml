type 'a tensor =
  | Zero of 'a
  | Pos of 'a tensor_record

and 'a tensor_record = {
  dims : int list;
  ndarray : 'a tensor Array.t;
}

type ('a, 'b) tensor_exception =
  | BadTensor of 'a tensor
  | BadTensorDims of 'a tensor list
  | BadTensorDims2 of 'a tensor * 'b tensor
  | BadIndex of 'a tensor * int list
  | BadTensorList of 'a tensor list

let raise_tensor_exn (type a) (type b) tnsr_exn =
  match tnsr_exn with
  | BadTensor x ->
      let exception BadTensor of a tensor in
      raise (BadTensor x)
  | BadTensorDims x ->
      let exception BadTensorDims of a tensor list in
      raise (BadTensorDims x)
  | BadTensorDims2 (x, y) ->
      let exception BadTensorDims2 of a tensor * b tensor in
      raise (BadTensorDims2 (x, y))
  | BadIndex (x, l) ->
      let exception BadIndex of a tensor * int list in
      raise (BadIndex (x, l))
  | BadTensorList x ->
      let exception BadTensorList of a tensor list in
      raise (BadTensorList x)

let rec check_dims_rec tnsr dimslist =
  match tnsr with
  | Zero x -> true
  | Pos { dims; ndarray } -> (
      match dims with
      | [] -> false
      | l :: tdims ->
          let dims_match =
            match dimslist with
            | None -> true
            | Some x -> x = dims
          in
          dims_match
          && l = Array.length ndarray
          && Array.for_all (fun tn -> check_dims_rec tn (Some tdims)) ndarray)

let check_dims tnsr =
  if check_dims_rec tnsr None then () else raise_tensor_exn (BadTensor tnsr)

let string_of_int_list lst =
  "[" ^ String.concat "," (List.map string_of_int lst) ^ "]"

let rec string_of_tensor_rec tnsr print_dim =
  match tnsr with
  | Zero x ->
      (if print_dim then "dims: " ^ string_of_int_list [] ^ "\n" else "")
      ^ "[" ^ string_of_float x ^ "]"
  | Pos { dims; ndarray } ->
      (if print_dim then "dims: " ^ string_of_int_list dims ^ "\n" else "")
      ^ "["
      ^ (Array.map (fun t -> string_of_tensor_rec t false) ndarray
        |> Array.to_list |> String.concat ", ")
      ^ "]"

let rec string_of_tensor tnsr = string_of_tensor_rec tnsr true

let rec make_tensor_rec dims f acc =
  match dims with
  | [] -> Zero (f (List.rev acc))
  | l :: tdims ->
      Pos
        {
          dims;
          ndarray = Array.init l (fun i -> make_tensor_rec tdims f (i :: acc));
        }

let make_tensor dims f = make_tensor_rec dims f []
let zeroes dims = make_tensor_rec dims (fun _ -> 0.) []
let ones dims = make_tensor_rec dims (fun _ -> 1.) []
let random dims = make_tensor_rec dims (fun _ -> Random.float 1.) []

let same_dims tnsr1 tnsr2 =
  match (tnsr1, tnsr2) with
  | Zero x, Zero y -> true
  | Pos x, Pos y -> x.dims = y.dims
  | _ -> false

let rec pointwise f tnsr1 =
  match tnsr1 with
  | Zero x -> Zero (f x)
  | Pos x -> (
      match x.dims with
      | [] -> raise_tensor_exn (BadTensor tnsr1)
      | l :: tdims ->
          Pos
            {
              dims = x.dims;
              ndarray = Array.init l (fun i -> pointwise f x.ndarray.(i));
            })

let rec pointwise2 f tnsr1 tnsr2 =
  match (tnsr1, tnsr2) with
  | Zero x, Zero y -> Zero (f x y)
  | Pos x, Pos y -> (
      match x.dims with
      | [] -> raise_tensor_exn (BadTensor tnsr1)
      | l :: tdims ->
          Pos
            {
              dims = x.dims;
              ndarray =
                Array.init l (fun i -> pointwise2 f x.ndarray.(i) y.ndarray.(i));
            })
  | _ -> raise_tensor_exn (BadTensorDims2 (tnsr1, tnsr2))

let rec all_zero tnsr_lst =
  match tnsr_lst with
  | [] -> true
  | tnsr :: tail -> (
      match tnsr with
      | Zero x -> all_zero tail
      | _ -> false)

let rec all_pos tnsr_lst =
  match tnsr_lst with
  | [] -> true
  | tnsr :: tail -> (
      match tnsr with
      | Pos x -> all_pos tail
      | _ -> false)

let rec unbox_zero tnsr_lst =
  List.map
    (fun tnsr ->
      match tnsr with
      | Zero x -> x
      | Pos x -> raise_tensor_exn (BadTensor tnsr))
    tnsr_lst

let rec unbox_pos tnsr_lst =
  List.map
    (fun tnsr ->
      match tnsr with
      | Zero x -> raise_tensor_exn (BadTensor tnsr)
      | Pos x -> x)
    tnsr_lst

let get_nth_tensor_record n tnsr_rcrd_lst =
  List.map (fun tnsr_rcrd -> tnsr_rcrd.ndarray.(n)) tnsr_rcrd_lst

let rec pointwisen f tnsr_lst =
  if all_zero tnsr_lst then Zero (f (unbox_zero tnsr_lst))
  else if all_pos tnsr_lst then
    let tnsr_rcrd_lst = unbox_pos tnsr_lst in
    match tnsr_rcrd_lst with
    | [] -> raise (failwith "Impossible")
    | x :: tnsr_tail -> (
        match x.dims with
        | [] -> raise_tensor_exn (BadTensor (Pos x))
        | l :: tdims ->
            Pos
              {
                dims = x.dims;
                ndarray =
                  Array.init l (fun i ->
                      pointwisen f (get_nth_tensor_record i tnsr_rcrd_lst));
              })
  else raise_tensor_exn (BadTensorDims tnsr_lst)

let get_dims tnsr =
  match tnsr with
  | Zero x -> []
  | Pos x -> x.dims

let get_first_other_dims tnsr =
  match get_dims tnsr with
  | [] -> raise_tensor_exn (BadTensorDims [ tnsr ])
  | dim :: dim_tail -> (dim, dim_tail)

let wrap tnsr = Pos { dims = 1 :: get_dims tnsr; ndarray = [| tnsr |] }
let rec wrap_k tnsr k = if k = 0 then tnsr else wrap_k (wrap tnsr) (k - 1)

let rec flatten_tensor_array tnsr_array =
  Array.to_list tnsr_array
  |> List.map (fun tnsr ->
      match tnsr with
      | Zero x -> [| tnsr |]
      | Pos x -> flatten_tensor_array x.ndarray)
  |> Array.concat

let rec flatten tnsr =
  match tnsr with
  | Zero x -> Pos { dims = [ 1 ]; ndarray = [| tnsr |] }
  | Pos x ->
      let ndarray = flatten_tensor_array x.ndarray in
      Pos { dims = [ Array.length ndarray ]; ndarray }

let int_dot v1 v2 = List.fold_left2 (fun acc x1 x2 -> acc + (x1 * x2)) 0 v1 v2

let float_dot v1 v2 =
  List.fold_left2 (fun acc x1 x2 -> acc +. (x1 *. x2)) 0. v1 v2

let get_ndarray tnsr =
  match tnsr with
  | Zero x -> [| tnsr |]
  | Pos x -> x.ndarray

let rec lookup tnsr indx_lst =
  match (tnsr, indx_lst) with
  | Zero x, [] -> x
  | Pos x, i :: indx_tail -> lookup x.ndarray.(i) indx_tail
  | _ -> raise_tensor_exn (BadIndex (tnsr, indx_lst))

let rec flattened_lookup flat_tnsr tnsr_dims indx_lst =
  let size, offsets =
    List.fold_right
      (fun dim (cur_offset, acc) -> (dim * cur_offset, cur_offset :: acc))
      tnsr_dims (1, [])
  in
  let indx = int_dot offsets indx_lst in
  match (get_ndarray flat_tnsr).(indx) with
  | Zero x -> x
  | Pos x -> raise_tensor_exn (BadIndex (flat_tnsr, [ indx ]))

let rec insert_k lst k x =
  if k = 0 then x :: lst
  else
    match lst with
    | [] -> [ x ]
    | h :: t -> h :: insert_k t (k - 1) x

let rec delete_k lst k =
  match lst with
  | [] -> []
  | h :: t -> if k = 0 then t else h :: delete_k t (k - 1)

let rec set_k lst k x = insert_k (delete_k lst k) k x

let rec split_k_rec lst k acc =
  match lst with
  | [] -> (List.rev acc, [])
  | h :: t ->
      if k = 0 then (List.rev acc, lst) else split_k_rec t (k - 1) (h :: acc)

let split_k lst k = split_k_rec lst k []

let contract tnsr1 tnsr2 indx1 indx2 =
  let dims1 = get_dims tnsr1 in
  let dims2 = get_dims tnsr2 in
  let l = List.nth dims1 indx1 in
  make_tensor
    (delete_k dims1 indx1 @ delete_k dims2 indx2)
    (fun indx_lst ->
      let indx_lst1, indx_lst2 = split_k indx_lst (List.length dims1 - 1) in
      let v1 =
        List.init l (fun i -> lookup tnsr1 (insert_k indx_lst1 indx1 i))
      in
      let v2 =
        List.init l (fun i -> lookup tnsr2 (insert_k indx_lst2 indx2 i))
      in
      float_dot v1 v2)

let matmul tnsr1 tnsr2 = contract tnsr1 tnsr2 1 0

let get_single i_lst =
  match i_lst with
  | h :: _ -> h
  | _ -> raise (failwith "not single")

let tensor_of_float flt = Zero flt

let tensor_of_1d_array arr =
  make_tensor [ Array.length arr ] (fun i -> arr.(get_single i))

let tensor_of_1d_list lst =
  make_tensor [ List.length lst ] (fun i -> List.nth lst (get_single i))

let stack_list tnsr_lst =
  match tnsr_lst with
  | [] -> raise_tensor_exn (BadTensorList tnsr_lst)
  | tnsr :: _ ->
      let l = List.length tnsr_lst in
      Pos
        {
          dims = l :: get_dims tnsr;
          ndarray = Array.init l (fun i -> List.nth tnsr_lst i);
        }

let stack_array tnsr_arr =
  let l = Array.length tnsr_arr in
  Pos
    {
      dims = l :: get_dims tnsr_arr.(0);
      ndarray = Array.init l (fun i -> tnsr_arr.(i));
    }

let tensor_of_2d_list lst_mtrx =
  stack_list (List.map (fun lst -> tensor_of_1d_list lst) lst_mtrx)

let tensor_of_2d_array arr_mtrx =
  stack_array (Array.map (fun arr -> tensor_of_1d_array arr) arr_mtrx)

let swap lst i j =
  let xi = List.nth lst i in
  let xj = List.nth lst j in
  let lst' = set_k lst i xj in
  set_k lst' j xi

let permute lst prm =
  List.init (List.length prm) (fun i -> List.nth lst (List.nth prm i))

let swap_dims tnsr i j =
  match tnsr with
  | Zero x -> tnsr
  | Pos x ->
      make_tensor (swap x.dims i j) (fun indx_lst ->
          lookup tnsr (swap indx_lst i j))

let permute_dims tnsr prm =
  match tnsr with
  | Zero x -> tnsr
  | Pos x ->
      make_tensor (permute x.dims prm) (fun indx_lst ->
          lookup tnsr (permute indx_lst prm))

let group tnsr k =
  match tnsr with
  | Zero x -> tnsr
  | Pos x ->
      stack_array
        (Array.init
           (Array.length x.ndarray / k)
           (fun i -> stack_array (Array.sub x.ndarray (k * i) k)))

let rec group_rec tnsr dims =
  match dims with
  | [] -> tnsr
  | dim :: dim_tail -> group_rec (group tnsr dim) dim_tail

let group_dims tnsr dims = group_rec tnsr (List.rev dims)

let reshape tnsr dims =
  let tnsr' = flatten tnsr in
  let tnsr'' = group_dims tnsr' dims in
  match tnsr'' with
  | Zero x -> raise_tensor_exn (BadTensor tnsr)
  | Pos x -> x.ndarray.(0)

let flatten_one tnsr =
  match tnsr with
  | Zero x -> Pos { dims = [ 1 ]; ndarray = [| tnsr |] }
  | Pos x -> stack_array x.ndarray

let rec flatten_k tnsr k =
  if k = 0 then tnsr else flatten_k (flatten_one tnsr) (k - 1)
