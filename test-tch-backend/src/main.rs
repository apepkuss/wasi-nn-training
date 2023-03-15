use mnist::*;
use ndarray::{Array, Array2, Array3};

mod plugin {
    #[link(wasm_import_module = "naive-math")]
    extern "C" {
        pub fn add(x: i32, y: i32) -> i32;
        pub fn set_input_tensor(
            data_offset: i32,
            data_size: i32,
            dims_offset: i32,
            dims_size: i32,
            ty: i32,
        );
        pub fn train(
            train_images_offset: i32,
            train_images_size: i32,
            train_labels_offset: i32,
            train_labels_size: i32,
            test_images_offset: i32,
            test_images_size: i32,
            test_labels_offset: i32,
            test_labels_size: i32,
            labels: i64,
            device: i32,
        );
    }
}

use byteorder::{BigEndian, ReadBytesExt};
use std::any::Any;
use std::io::prelude::*;
use std::path::Path;

const TRAIN_SIZE: usize = 50_000;
const VAL_SIZE: usize = 10_000;
const TEST_SIZE: usize = 10_000;

static TRN_LEN: u32 = 60_000;
static TST_LEN: u32 = 10_000;

const HEIGHT: usize = 28;
const WIDTH: usize = 28;

static TRN_IMG_FILENAME: &str = "train-images-idx3-ubyte";
static TRN_LBL_FILENAME: &str = "train-labels-idx1-ubyte";
static TST_IMG_FILENAME: &str = "t10k-images-idx3-ubyte";
static TST_LBL_FILENAME: &str = "t10k-labels-idx1-ubyte";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let x: i32 = args[1].parse().unwrap();
    let y: i32 = args[2].parse().unwrap();

    let res = unsafe { plugin::add(x, y) };
    println!("{x} + {y} = {res}");

    // use walkdir::WalkDir;

    // let current_dir = std::env::current_dir().expect("failed to get current_dir");
    // println!("current dir: {:?}", current_dir);
    // let path = current_dir.as_path();

    // for entry in WalkDir::new(path) {
    //     let entry = entry.expect("failed to get entry");
    //     println!("{}", entry.path().display());
    // }

    {
        let trn_img_filename = Path::new("data/train-images-idx3-ubyte");
        let mut trn_img = images(trn_img_filename, TRN_LEN);

        let trn_lbl_filename = Path::new("data/train-labels-idx1-ubyte");
        let mut trn_lbl = labels(trn_lbl_filename, TRN_LEN);

        let tst_img_filename = Path::new("data/t10k-images-idx3-ubyte");
        let mut tst_img = images(tst_img_filename, TST_LEN);

        let tst_lbl_filename = Path::new("data/t10k-labels-idx1-ubyte");
        let mut tst_lbl = labels(tst_lbl_filename, TST_LEN);

        let trn_len = TRAIN_SIZE;
        let val_len = VAL_SIZE;
        let tst_len = TEST_SIZE;

        trn_img.append(&mut tst_img);
        trn_lbl.append(&mut tst_lbl);
        let mut val_img = trn_img.split_off(trn_len * ROWS * COLS);
        let mut val_lbl = trn_lbl.split_off(trn_len);
        let mut tst_img = val_img.split_off(val_len * ROWS * COLS);
        let mut tst_lbl = val_lbl.split_off(val_len);
        tst_img.split_off(tst_len * ROWS * COLS);
        tst_lbl.split_off(tst_len);

        println!("trn_img len: {}", trn_img.len());
        println!("trn_lbl len: {}", trn_lbl.len());
        println!("tst_img len: {}", tst_img.len());
        println!("tst_lbl len: {}", tst_lbl.len());

        let train_images: Array<f32, _> = image_to_ndarray(trn_img, TRAIN_SIZE, HEIGHT, WIDTH);
        let train_labels: Array<i64, _> = labels_to_ndarray(trn_lbl, TRAIN_SIZE, 1);
        let test_images: Array<f32, _> = image_to_ndarray(tst_img, TEST_SIZE, HEIGHT, WIDTH);
        let test_labels: Array<i64, _> = labels_to_ndarray(tst_lbl, TEST_SIZE, 1);

        // train images
        // let train_images: ndarray::Array<f32, _> =
        //     ndarray::array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
        let train_images_dims: Vec<i32> = train_images.shape().iter().map(|&c| c as i32).collect();
        let train_images_dims_bytes = protocol::to_bytes(train_images_dims.as_slice());
        let train_images_slice = train_images
            .as_slice()
            .expect("failed to convert ndarray to slice");
        let train_images_bytes = protocol::to_bytes(train_images_slice);
        println!("size of train_images_bytes: {}", train_images_bytes.len());

        // train labels
        // let train_labels: ndarray::Array<f32, _> =
        //     ndarray::array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
        let train_labels_dims: Vec<i32> = train_labels.shape().iter().map(|&c| c as i32).collect();
        let train_labels_dims_bytes = protocol::to_bytes(train_labels_dims.as_slice());
        let train_labels_slice = train_labels
            .as_slice()
            .expect("failed to convert ndarray to slice");
        let train_labels_bytes = protocol::to_bytes(train_labels_slice);
        println!("size of train_labels_bytes: {}", train_labels_bytes.len());

        // test images
        // let test_images: ndarray::Array<f32, _> =
        //     ndarray::array![[1., 2., 3., 4.], [2., 3., 4., 1.], [3., 4., 1., 2.]];
        let test_images_dims: Vec<i32> = test_images.shape().iter().map(|&c| c as i32).collect();
        let test_images_dims_bytes = protocol::to_bytes(test_images_dims.as_slice());
        let test_images_slice = test_images
            .as_slice()
            .expect("failed to convert ndarray to slice");
        let test_images_bytes = protocol::to_bytes(test_images_slice);
        println!("size of test_images_bytes: {}", test_images_bytes.len());

        // test labels
        // let test_labels: ndarray::Array<f32, _> =
        //     ndarray::array![[1., 2., 3., 4.], [2., 3., 4., 1.], [3., 4., 1., 2.]];
        let test_labels_dims: Vec<i32> = test_labels.shape().iter().map(|&c| c as i32).collect();
        let test_labels_dims_bytes = protocol::to_bytes(test_labels_dims.as_slice());
        let test_labels_slice = test_labels
            .as_slice()
            .expect("failed to convert ndarray to slice");
        let test_labels_bytes = protocol::to_bytes(test_labels_slice);
        println!("size of test_labels_bytes: {}", test_labels_bytes.len());
    }

    // train images
    let train_images: ndarray::Array<f32, _> =
        ndarray::array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
    let train_images_dims: Vec<i32> = train_images.shape().iter().map(|&c| c as i32).collect();
    let train_images_dims_bytes = protocol::to_bytes(train_images_dims.as_slice());
    let train_images_slice = train_images
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let train_images_bytes = protocol::to_bytes(train_images_slice);
    println!("size of train_images_bytes: {}", train_images_bytes.len());

    // train labels
    let train_labels: ndarray::Array<f32, _> =
        ndarray::array![[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]];
    let train_labels_dims: Vec<i32> = train_labels.shape().iter().map(|&c| c as i32).collect();
    let train_labels_dims_bytes = protocol::to_bytes(train_labels_dims.as_slice());
    let train_labels_slice = train_labels
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let train_labels_bytes = protocol::to_bytes(train_labels_slice);
    println!("size of train_labels_bytes: {}", train_labels_bytes.len());

    // test images
    let test_images: ndarray::Array<f32, _> =
        ndarray::array![[1., 2., 3., 4.], [2., 3., 4., 1.], [3., 4., 1., 2.]];
    let test_images_dims: Vec<i32> = test_images.shape().iter().map(|&c| c as i32).collect();
    let test_images_dims_bytes = protocol::to_bytes(test_images_dims.as_slice());
    let test_images_slice = test_images
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let test_images_bytes = protocol::to_bytes(test_images_slice);
    println!("size of test_images_bytes: {}", test_images_bytes.len());

    // test labels
    let test_labels: ndarray::Array<f32, _> =
        ndarray::array![[1., 2., 3., 4.], [2., 3., 4., 1.], [3., 4., 1., 2.]];
    let test_labels_dims: Vec<i32> = test_labels.shape().iter().map(|&c| c as i32).collect();
    let test_labels_dims_bytes = protocol::to_bytes(test_labels_dims.as_slice());
    let test_labels_slice = test_labels
        .as_slice()
        .expect("failed to convert ndarray to slice");
    let test_labels_bytes = protocol::to_bytes(test_labels_slice);
    println!("size of test_labels_bytes: {}", test_labels_bytes.len());

    // println!("size of Array: {}", std::mem::size_of::<protocol::Array>());
    // println!("size of *const u8: {}", std::mem::size_of::<*const u8>());
    // println!("size of u8: {}", std::mem::size_of::<u8>());

    // let nums = [1u8, 2, 3];
    // let ptr = &nums as *const u8;
    // println!("ptr address: {:p}", ptr);
    // let offset = ptr as i32;
    // println!("offset: {}", offset);
    // unsafe { plugin::set_input_tensor(offset, 3) };

    // =======
    {
        // println!("size of Array: {}", std::mem::size_of::<protocol::Array>());

        // let data1 = [1u8, 2, 3, 2, 3, 1, 3, 1, 2];
        // let arr1 = protocol::Array {
        //     data: &data1 as *const _,
        //     size: data1.len() as i32,
        // };
        // // let data2 = [6u8, 7, 8];
        // // let arr2 = protocol::Array {
        // //     data: &data2 as *const _,
        // //     size: data2.len() as i32,
        // // };

        // let arrs = [arr1];
        // let ptr = &arrs as *const _ as usize as i32;

        // let dims1 = [3u8, 3];
        // let dims1_ptr = &dims1 as *const _ as usize as i32;
        // let dims1_size = dims1.len();

        // unsafe {
        //     plugin::set_input_tensor(
        //         ptr,
        //         (arrs.len() * std::mem::size_of::<protocol::Array>()) as i32,
        //         dims1_ptr,
        //         dims1_size as i32,
        //         2,
        //     );
        // }
    }
    // ======

    println!(
        "size of MyTensor: {}",
        std::mem::size_of::<protocol::MyTensor>()
    );

    let train_images = protocol::MyTensor {
        data: train_images_bytes.as_ptr() as *const _,
        data_size: train_images_bytes.len() as u32,
        dims: train_images_dims_bytes.as_ptr() as *const _,
        dims_size: train_images_dims_bytes.len() as u32,
        ty: 1,
    };
    let train_images_offset = &train_images as *const _ as usize as i32;
    let train_images_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    let train_labels = protocol::MyTensor {
        data: train_labels_bytes.as_ptr() as *const _,
        data_size: train_labels_bytes.len() as u32,
        dims: train_labels_dims_bytes.as_ptr() as *const _,
        dims_size: train_labels_dims_bytes.len() as u32,
        ty: 1,
    };
    let train_labels_offset = &train_labels as *const _ as usize as i32;
    let train_labels_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    let test_images = protocol::MyTensor {
        data: test_images_bytes.as_ptr() as *const _,
        data_size: test_images_bytes.len() as u32,
        dims: test_images_dims_bytes.as_ptr() as *const _,
        dims_size: test_images_dims_bytes.len() as u32,
        ty: 1,
    };
    let test_images_offset = &test_images as *const _ as usize as i32;
    let test_images_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    let test_labels = protocol::MyTensor {
        data: test_labels_bytes.as_ptr() as *const _,
        data_size: test_labels_bytes.len() as u32,
        dims: test_labels_dims_bytes.as_ptr() as *const _,
        dims_size: test_labels_dims_bytes.len() as u32,
        ty: 1,
    };
    let test_labels_offset = &test_labels as *const _ as usize as i32;
    let test_labels_size = std::mem::size_of::<protocol::MyTensor>() as i32;

    unsafe {
        plugin::train(
            train_images_offset,
            train_images_size,
            train_labels_offset,
            train_labels_size,
            test_images_offset,
            test_images_size,
            test_labels_offset,
            test_labels_size,
            10,
            0, // device: CPU
        )
    }
}

static LBL_MAGIC_NUMBER: u32 = 0x0000_0801;

fn labels(path: &std::path::Path, expected_length: u32) -> Vec<u8> {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|_| panic!("Unable to find path to labels at {:?}.", path));
    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(
        LBL_MAGIC_NUMBER == magic_number,
        "Expected magic number {} got {}.",
        LBL_MAGIC_NUMBER,
        magic_number
    );
    let length = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
    assert!(
        expected_length == length,
        "Expected data set length of {} got {}.",
        expected_length,
        length
    );
    file.bytes().map(|b| b.unwrap()).collect()
}

static IMG_MAGIC_NUMBER: u32 = 0x0000_0803;
static ROWS: usize = 28;
static COLS: usize = 28;

fn images(path: &std::path::Path, expected_length: u32) -> Vec<u8> {
    // Read whole file in memory
    let mut content: Vec<u8> = Vec::new();
    let mut file = {
        let mut fh = std::fs::File::open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to images at {:?}.", path));
        let _ = fh
            .read_to_end(&mut content)
            .unwrap_or_else(|_| panic!("Unable to read whole file in memory ({})", path.display()));
        // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
        // used with a `Vec` directly, it requires a slice.
        &content[..]
    };

    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(
        IMG_MAGIC_NUMBER == magic_number,
        "Expected magic number {} got {}.",
        IMG_MAGIC_NUMBER,
        magic_number
    );
    let length = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
    assert!(
        expected_length == length,
        "Expected data set length of {} got {}.",
        expected_length,
        length
    );
    let rows = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to number of rows from {:?}.", path))
        as usize;
    assert!(
        ROWS == rows,
        "Expected rows length of {} got {}.",
        ROWS,
        rows
    );
    let cols = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to number of columns from {:?}.", path))
        as usize;
    assert!(
        COLS == cols,
        "Expected cols length of {} got {}.",
        COLS,
        cols
    );
    // Convert `file` from a Vec to a slice.
    file.to_vec()
}

pub fn image_to_ndarray(data: Vec<u8>, dim1: usize, dim2: usize, dim3: usize) -> Array3<f32> {
    // normalize the image as well
    let inp_data: Array3<f32> = Array3::from_shape_vec((dim1, dim2, dim3), data)
        .expect("Error converting data to 3D array")
        .map(|x| *x as f32 / 256.0);

    inp_data
}

pub fn labels_to_ndarray(data: Vec<u8>, dim1: usize, dim2: usize) -> Array2<i64> {
    let inp_data: Array2<i64> = Array2::from_shape_vec((dim1, dim2), data)
        .expect("Error converting data to 2D array")
        .map(|x| *x as i64);

    inp_data
}

// * interface protocol

pub mod protocol {

    use byteorder::{LittleEndian, ReadBytesExt};
    use std::io::Cursor;

    pub fn to_bytes<'a, T>(data: &'a [T]) -> &'a [u8] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const _,
                data.len() * std::mem::size_of::<T>(),
            )
        }
    }

    pub fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
        let chunks: Vec<&[u8]> = data.chunks(4).collect();
        let v: Vec<f32> = chunks
            .into_iter()
            .map(|c| {
                let mut rdr = Cursor::new(c);
                rdr.read_f32::<LittleEndian>().expect("failed to read")
            })
            .collect();

        v.into_iter().collect()
    }

    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct Array {
        pub data: *const u8, // 4 bytes
        pub size: i32,       // 4 bytes
    }

    #[repr(C)]
    #[derive(Clone, Debug)]
    pub struct MyTensor {
        pub data: *const u8, // 4 bytes
        pub data_size: u32,  // 4 bytes
        pub dims: *const u8, // 4 bytes
        pub dims_size: u32,  // 4 bytes
        pub ty: u8,          // 1 byte
    }

    pub enum Device {
        Cpu,
        Cuda(usize),
        Mps,
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug)]
    pub struct Tensor<'a> {
        pub dimensions: TensorDimensions<'a>, // 8 bytes
        pub ty: TensorType,                   // 1 bytes
        pub data: TensorData<'a>,             // 8 bytes
    }

    pub type TensorData<'a> = &'a [u8];
    pub type TensorDimensions<'a> = &'a [u32];

    #[repr(transparent)]
    #[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
    pub struct TensorType(u8);
    pub const TENSOR_TYPE_F16: TensorType = TensorType(0);
    pub const TENSOR_TYPE_F32: TensorType = TensorType(1);
    pub const TENSOR_TYPE_U8: TensorType = TensorType(2);
    pub const TENSOR_TYPE_I32: TensorType = TensorType(3);
    impl TensorType {
        pub const fn raw(&self) -> u8 {
            self.0
        }

        pub fn name(&self) -> &'static str {
            match self.0 {
                0 => "F16",
                1 => "F32",
                2 => "U8",
                3 => "I32",
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }
        pub fn message(&self) -> &'static str {
            match self.0 {
                0 => "",
                1 => "",
                2 => "",
                3 => "",
                _ => unsafe { core::hint::unreachable_unchecked() },
            }
        }
    }
    impl std::fmt::Debug for TensorType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TensorType")
                .field("code", &self.0)
                .field("name", &self.name())
                .field("message", &self.message())
                .finish()
        }
    }
}
