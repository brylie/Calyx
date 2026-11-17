//! Adversarial Challenge Test Suite for Calyx DSP Math Primitives (`Matrix` and `Tensor`).
//!
//! Evaluates algebraic invariants, multilinear algebra identities, dimension handling,
//! boundary conditions, and stress scenarios.

use calyx_dsp::error::DspError;
use calyx_dsp::math::{Matrix, Tensor};

const EPSILON: f64 = 1e-12;

// ============================================================================
// 1. Multilinear Algebra Identities: Fold/Unfold Lossless Roundtrip
// ============================================================================

#[test]
fn challenge_fold_unfold_lossless_roundtrip_all_dimensions() {
    let test_shapes: Vec<Vec<usize>> = vec![
        vec![1],                        // 1D scalar
        vec![7],                        // 1D vector
        vec![1, 1],                     // 2D scalar
        vec![4, 9],                     // 2D matrix
        vec![1, 1, 1],                  // 3D singletons
        vec![3, 4, 5],                  // 3D general
        vec![5, 1, 4],                  // 3D with singleton
        vec![2, 3, 4, 5],               // 4D general
        vec![1, 4, 1, 3],               // 4D with multiple singletons
        vec![2, 3, 2, 4, 2],            // 5D hypercube
        vec![3, 2, 5, 2, 3],            // 5D asymmetric
        vec![2, 2, 2, 2, 2, 2],         // 6D
        vec![2, 1, 2, 1, 2, 1, 2],      // 7D with alternating singletons
    ];

    for shape in test_shapes {
        let ndim = shape.len();
        // Deterministic pseudo-data generator based on multi-index coordinates
        let tensor = Tensor::from_shape_fn(shape.clone(), |idx| {
            idx.iter()
                .enumerate()
                .map(|(i, &x)| (x + 1) as f64 * (i + 1) as f64 * std::f64::consts::PI)
                .sum::<f64>()
        })
        .unwrap();

        for mode in 0..ndim {
            let unfolded = tensor
                .mode_unfold(mode)
                .unwrap_or_else(|e| panic!("mode_unfold failed on mode {} of shape {:?}: {:?}", mode, shape, e));

            // Verify dimensions of unfolded matrix
            assert_eq!(unfolded.rows, shape[mode]);
            let expected_cols: usize = shape.iter().enumerate().filter(|&(k, _)| k != mode).map(|(_, &d)| d).product();
            assert_eq!(unfolded.cols, expected_cols);

            let folded = Tensor::fold(&unfolded, mode, &shape)
                .unwrap_or_else(|e| panic!("fold failed on mode {} of shape {:?}: {:?}", mode, shape, e));

            assert_eq!(folded.shape(), tensor.shape());
            assert_eq!(folded.strides(), tensor.strides());
            assert_eq!(folded.len(), tensor.len());

            for (i, (&actual, &expected)) in folded.data().iter().zip(tensor.data()).enumerate() {
                assert!(
                    (actual - expected).abs() < EPSILON,
                    "Roundtrip mismatch at index {} for shape {:?}, mode {}: got {}, expected {}",
                    i,
                    shape,
                    mode,
                    actual,
                    expected
                );
            }
        }
    }
}

// ============================================================================
// 2. Multilinear Algebra Identities: Mode Product Commutativity
// ============================================================================

#[test]
fn challenge_mode_product_commutativity_3d_4d_5d() {
    // Identity: X x_p A x_q B == X x_q B x_p A for all distinct modes p != q.
    // Must hold for rectangular (non-square) matrices A and B.

    // 3D Tensor: shape [3, 4, 5]
    let t3 = Tensor::from_shape_fn(vec![3, 4, 5], |idx| {
        (idx[0] * 31 + idx[1] * 7 + idx[2] * 2 + 1) as f64
    })
    .unwrap();

    let m_3_to_2 = Matrix::from_shape_fn(2, 3, |r, c| (r * 2 + c + 1) as f64).unwrap();
    let m_4_to_6 = Matrix::from_shape_fn(6, 4, |r, c| (r + c * 3 + 1) as f64).unwrap();
    let m_5_to_3 = Matrix::from_shape_fn(3, 5, |r, c| (r * 4 + c + 2) as f64).unwrap();

    let matrices_3d = [m_3_to_2, m_4_to_6, m_5_to_3];

    for p in 0..3 {
        for q in 0..3 {
            if p == q {
                continue;
            }
            let a = &matrices_3d[p];
            let b = &matrices_3d[q];

            let prod_ab = t3.mode_product(p, a).unwrap().mode_product(q, b).unwrap();
            let prod_ba = t3.mode_product(q, b).unwrap().mode_product(p, a).unwrap();

            assert_eq!(prod_ab.shape(), prod_ba.shape(), "Shape mismatch for p={}, q={}", p, q);
            for (i, (&val_ab, &val_ba)) in prod_ab.data().iter().zip(prod_ba.data()).enumerate() {
                assert!(
                    (val_ab - val_ba).abs() < EPSILON,
                    "Commutativity failed at elem {} for modes p={}, q={}: {} vs {}",
                    i,
                    p,
                    q,
                    val_ab,
                    val_ba
                );
            }
        }
    }

    // 4D Tensor: shape [2, 3, 4, 2]
    let t4 = Tensor::from_shape_fn(vec![2, 3, 4, 2], |idx| {
        (idx[0] * 13 + idx[1] * 11 + idx[2] * 5 + idx[3] * 3 + 1) as f64
    })
    .unwrap();

    let matrices_4d = [
        Matrix::from_shape_fn(4, 2, |r, c| (r + c + 1) as f64).unwrap(), // mode 0: 2 -> 4
        Matrix::from_shape_fn(5, 3, |r, c| (r * 2 + c) as f64).unwrap(), // mode 1: 3 -> 5
        Matrix::from_shape_fn(2, 4, |r, c| (r + c * 2) as f64).unwrap(), // mode 2: 4 -> 2
        Matrix::from_shape_fn(3, 2, |r, c| (r * 3 + c) as f64).unwrap(), // mode 3: 2 -> 3
    ];

    for p in 0..4 {
        for q in 0..4 {
            if p == q {
                continue;
            }
            let a = &matrices_4d[p];
            let b = &matrices_4d[q];

            let prod_ab = t4.mode_product(p, a).unwrap().mode_product(q, b).unwrap();
            let prod_ba = t4.mode_product(q, b).unwrap().mode_product(p, a).unwrap();

            assert_eq!(prod_ab.shape(), prod_ba.shape(), "4D shape mismatch for p={}, q={}", p, q);
            for (i, (&val_ab, &val_ba)) in prod_ab.data().iter().zip(prod_ba.data()).enumerate() {
                assert!(
                    (val_ab - val_ba).abs() < EPSILON,
                    "4D commutativity failed at elem {} for p={}, q={}: {} vs {}",
                    i,
                    p,
                    q,
                    val_ab,
                    val_ba
                );
            }
        }
    }

    // 5D Tensor: shape [2, 3, 2, 4, 2]
    let t5 = Tensor::from_shape_fn(vec![2, 3, 2, 4, 2], |idx| {
        (idx[0] * 7 + idx[1] * 5 + idx[2] * 3 + idx[3] * 2 + idx[4] + 1) as f64
    })
    .unwrap();

    let matrices_5d = [
        Matrix::from_shape_fn(3, 2, |r, c| (r + c + 1) as f64).unwrap(),
        Matrix::from_shape_fn(2, 3, |r, c| (r * 2 + c) as f64).unwrap(),
        Matrix::from_shape_fn(4, 2, |r, c| (r + c * 2) as f64).unwrap(),
        Matrix::from_shape_fn(2, 4, |r, c| (r * 3 + c) as f64).unwrap(),
        Matrix::from_shape_fn(5, 2, |r, c| (r + c + 3) as f64).unwrap(),
    ];

    for p in 0..5 {
        for q in 0..5 {
            if p == q {
                continue;
            }
            let a = &matrices_5d[p];
            let b = &matrices_5d[q];

            let prod_ab = t5.mode_product(p, a).unwrap().mode_product(q, b).unwrap();
            let prod_ba = t5.mode_product(q, b).unwrap().mode_product(p, a).unwrap();

            assert_eq!(prod_ab.shape(), prod_ba.shape(), "5D shape mismatch for p={}, q={}", p, q);
            for (i, (&val_ab, &val_ba)) in prod_ab.data().iter().zip(prod_ba.data()).enumerate() {
                assert!(
                    (val_ab - val_ba).abs() < EPSILON,
                    "5D commutativity failed at elem {} for p={}, q={}: {} vs {}",
                    i,
                    p,
                    q,
                    val_ab,
                    val_ba
                );
            }
        }
    }
}

#[test]
fn challenge_mode_product_commutativity_3_modes_all_permutations() {
    // Identity across 3 distinct modes in 5D tensor:
    // Any permutation of applying mode products on modes p, q, r must yield identical results.
    let t = Tensor::from_shape_fn(vec![2, 3, 2, 4, 2], |idx| {
        (idx[0] * 11 + idx[1] * 7 + idx[2] * 5 + idx[3] * 3 + idx[4] + 1) as f64
    })
    .unwrap();

    let a = Matrix::from_shape_fn(4, 2, |r, c| (r + c + 1) as f64).unwrap(); // mode 0
    let b = Matrix::from_shape_fn(5, 3, |r, c| (r * 2 + c) as f64).unwrap(); // mode 1
    let c = Matrix::from_shape_fn(3, 4, |r, c| (r + c * 2) as f64).unwrap(); // mode 3

    // 6 permutations of applying (0, A), (1, B), (3, C):
    let p_013 = t.mode_product(0, &a).unwrap().mode_product(1, &b).unwrap().mode_product(3, &c).unwrap();
    let p_031 = t.mode_product(0, &a).unwrap().mode_product(3, &c).unwrap().mode_product(1, &b).unwrap();
    let p_103 = t.mode_product(1, &b).unwrap().mode_product(0, &a).unwrap().mode_product(3, &c).unwrap();
    let p_130 = t.mode_product(1, &b).unwrap().mode_product(3, &c).unwrap().mode_product(0, &a).unwrap();
    let p_301 = t.mode_product(3, &c).unwrap().mode_product(0, &a).unwrap().mode_product(1, &b).unwrap();
    let p_310 = t.mode_product(3, &c).unwrap().mode_product(1, &b).unwrap().mode_product(0, &a).unwrap();

    let permutations = [&p_031, &p_103, &p_130, &p_301, &p_310];
    for (idx_perm, perm) in permutations.iter().enumerate() {
        assert_eq!(p_013.shape(), perm.shape());
        for (i, (&val_ref, &val_p)) in p_013.data().iter().zip(perm.data()).enumerate() {
            assert!(
                (val_ref - val_p).abs() < EPSILON,
                "3-mode permutation {} failed at elem {}: {} vs {}",
                idx_perm,
                i,
                val_ref,
                val_p
            );
        }
    }
}

// ============================================================================
// 3. Multilinear Algebra Identities: Mode Product Associativity
// ============================================================================

#[test]
fn challenge_mode_product_associativity_3d_4d_5d() {
    // Identity: (X x_n A) x_n B == X x_n (B * A)
    // where A is J x I_n, B is K x J, and B * A is K x I_n.

    let test_shapes = vec![
        vec![3, 4, 5],
        vec![2, 3, 4, 2],
        vec![2, 3, 2, 4, 2],
    ];

    for shape in test_shapes {
        let ndim = shape.len();
        let tensor = Tensor::from_shape_fn(shape.clone(), |idx| {
            idx.iter().enumerate().map(|(i, &x)| (x + 1) as f64 * (i + 1) as f64).sum::<f64>()
        })
        .unwrap();

        for mode in 0..ndim {
            let in_dim = shape[mode];
            let j_dim = in_dim + 2; // non-square A
            let k_dim = in_dim + 4; // non-square B

            let a = Matrix::from_shape_fn(j_dim, in_dim, |r, c| ((r + 1) * (c + 2)) as f64 * 0.1).unwrap();
            let b = Matrix::from_shape_fn(k_dim, j_dim, |r, c| ((r + 2) + (c + 1)) as f64 * 0.1).unwrap();

            // Chained mode products
            let chained = tensor.mode_product(mode, &a).unwrap().mode_product(mode, &b).unwrap();

            // Direct mode product with BA
            let ba = b.matmul(&a).unwrap();
            let direct = tensor.mode_product(mode, &ba).unwrap();

            assert_eq!(chained.shape(), direct.shape());
            let mut expected_shape = shape.clone();
            expected_shape[mode] = k_dim;
            assert_eq!(chained.shape(), expected_shape.as_slice());

            for (i, (&val_c, &val_d)) in chained.data().iter().zip(direct.data()).enumerate() {
                assert!(
                    (val_c - val_d).abs() < 1e-10,
                    "Associativity failed at index {} for shape {:?}, mode {}: chained {}, direct {}",
                    i,
                    shape,
                    mode,
                    val_c,
                    val_d
                );
            }
        }
    }
}

#[test]
fn challenge_mode_product_associativity_3_factors() {
    // ((X x_n A) x_n B) x_n C == X x_n (C * B * A)
    let t = Tensor::from_shape_fn(vec![2, 3, 2], |idx| {
        (idx[0] + idx[1] * 2 + idx[2] * 3 + 1) as f64
    })
    .unwrap();

    // Mode 1: initial dim is 3
    let a = Matrix::from_shape_fn(4, 3, |r, c| (r + c + 1) as f64 * 0.1).unwrap(); // 4 x 3
    let b = Matrix::from_shape_fn(5, 4, |r, c| (r * 2 + c) as f64 * 0.1).unwrap(); // 5 x 4
    let c = Matrix::from_shape_fn(2, 5, |r, c| (r + c * 2) as f64 * 0.1).unwrap(); // 2 x 5

    let chained = t
        .mode_product(1, &a)
        .unwrap()
        .mode_product(1, &b)
        .unwrap()
        .mode_product(1, &c)
        .unwrap();

    let cba = c.matmul(&b).unwrap().matmul(&a).unwrap();
    let direct = t.mode_product(1, &cba).unwrap();

    assert_eq!(chained.shape(), &[2, 2, 2]);
    assert_eq!(direct.shape(), &[2, 2, 2]);

    for (i, (&v_c, &v_d)) in chained.data().iter().zip(direct.data()).enumerate() {
        assert!((v_c - v_d).abs() < 1e-10, "3-factor associativity failed at {}: {} vs {}", i, v_c, v_d);
    }
}

// ============================================================================
// 4. Matricization Isometry & Norm Invariance
// ============================================================================

#[test]
fn challenge_matricization_frobenius_isometry() {
    // Multilinear invariant: ||X||_F == ||X_(n)||_F for all modes n.
    let shapes = vec![
        vec![5],
        vec![3, 7],
        vec![2, 3, 4],
        vec![3, 2, 4, 3],
        vec![2, 3, 2, 4, 2],
    ];

    for shape in shapes {
        let t = Tensor::from_shape_fn(shape.clone(), |idx| {
            idx.iter().enumerate().map(|(i, &x)| (x + 1) as f64 * (i + 1) as f64).sum::<f64>()
        })
        .unwrap();

        let tensor_norm = t.norm();

        for mode in 0..shape.len() {
            let unfolded = t.mode_unfold(mode).unwrap();
            let matrix_norm = unfolded.frobenius_norm();

            assert!(
                (tensor_norm - matrix_norm).abs() < EPSILON,
                "Frobenius isometry failed for shape {:?}, mode {}: tensor_norm={}, matrix_norm={}",
                shape,
                mode,
                tensor_norm,
                matrix_norm
            );
        }
    }
}

// ============================================================================
// 5. Identity Preservation
// ============================================================================

#[test]
fn challenge_mode_product_identity_preservation() {
    // X x_n I == X for all modes n
    let shapes = vec![
        vec![4],
        vec![3, 5],
        vec![2, 3, 4],
        vec![2, 2, 3, 2],
        vec![2, 3, 2, 2, 3],
    ];

    for shape in shapes {
        let tensor = Tensor::from_shape_fn(shape.clone(), |idx| {
            idx.iter().sum::<usize>() as f64 * 1.5 + 0.5
        })
        .unwrap();

        for mode in 0..shape.len() {
            let eye = Matrix::identity(shape[mode]).unwrap();
            let res = tensor.mode_product(mode, &eye).unwrap();
            assert_eq!(res.shape(), tensor.shape());
            for (i, (&actual, &expected)) in res.data().iter().zip(tensor.data()).enumerate() {
                assert!(
                    (actual - expected).abs() < EPSILON,
                    "Identity failed at {} for shape {:?}, mode {}",
                    i,
                    shape,
                    mode
                );
            }
        }
    }
}

// ============================================================================
// 6. Dimension Mismatch Rejection: Matmul and Mode Products
// ============================================================================

#[test]
fn challenge_dimension_mismatch_rejection_matmul() {
    let a = Matrix::zeros(3, 4).unwrap();
    let b = Matrix::zeros(5, 2).unwrap();
    match a.matmul(&b) {
        Err(DspError::DimensionMismatch { expected, actual }) => {
            assert_eq!(expected, vec![4]);
            assert_eq!(actual, vec![5]);
        }
        other => panic!("Expected DimensionMismatch, got {:?}", other),
    }

    let c = Matrix::zeros(2, 2).unwrap();
    let d = Matrix::zeros(3, 2).unwrap();
    assert!(c.matmul(&d).is_err());
}

#[test]
fn challenge_dimension_mismatch_rejection_mode_product() {
    let tensor = Tensor::zeros(vec![2, 3, 4]).unwrap();

    // Mode index out of bounds
    match tensor.mode_product(3, &Matrix::identity(4).unwrap()) {
        Err(DspError::InvalidParameter { name, .. }) => {
            assert_eq!(name, "mode");
        }
        other => panic!("Expected InvalidParameter for mode out of bounds, got {:?}", other),
    }

    // Matrix cols != tensor.shape[mode]
    let bad_mat = Matrix::zeros(5, 7).unwrap(); // cols = 7 != shape[1] = 3
    match tensor.mode_product(1, &bad_mat) {
        Err(DspError::DimensionMismatch { expected, actual }) => {
            assert_eq!(expected, vec![3]);
            assert_eq!(actual, vec![7]);
        }
        other => panic!("Expected DimensionMismatch for wrong matrix cols, got {:?}", other),
    }
}

#[test]
fn challenge_dimension_mismatch_rejection_fold() {
    // Mode out of bounds
    let m = Matrix::zeros(2, 12).unwrap();
    assert!(Tensor::fold(&m, 3, &[2, 3, 4]).is_err());

    // Wrong matrix rows
    let bad_rows = Matrix::zeros(3, 12).unwrap();
    assert!(Tensor::fold(&bad_rows, 0, &[2, 3, 4]).is_err());

    // Wrong matrix cols
    let bad_cols = Matrix::zeros(2, 11).unwrap();
    assert!(Tensor::fold(&bad_cols, 0, &[2, 3, 4]).is_err());
}

// ============================================================================
// 7. Stress Test: High Dimensions (8D) and Singletons
// ============================================================================

#[test]
fn challenge_high_dimensions_8d() {
    // 8-dimensional tensor: 2^8 = 256 elements
    let shape_8d = vec![2; 8];
    let tensor_8d = Tensor::from_shape_fn(shape_8d.clone(), |idx| {
        idx.iter().enumerate().map(|(i, &x)| x * (1 << i)).sum::<usize>() as f64
    })
    .unwrap();

    assert_eq!(tensor_8d.ndim(), 8);
    assert_eq!(tensor_8d.len(), 256);

    // Verify all 8 modes unfold and fold roundtrip
    for mode in 0..8 {
        let unfolded = tensor_8d.mode_unfold(mode).unwrap();
        assert_eq!(unfolded.rows, 2);
        assert_eq!(unfolded.cols, 128);

        let folded = Tensor::fold(&unfolded, mode, &shape_8d).unwrap();
        assert_eq!(folded, tensor_8d);
    }

    // Mode product on mode 0 and mode 7
    let shrink_mat = Matrix::from_vec(1, 2, vec![0.5, 0.5]).unwrap();
    let res_mode0 = tensor_8d.mode_product(0, &shrink_mat).unwrap();
    assert_eq!(res_mode0.shape(), &[1, 2, 2, 2, 2, 2, 2, 2]);

    let res_mode7 = tensor_8d.mode_product(7, &shrink_mat).unwrap();
    assert_eq!(res_mode7.shape(), &[2, 2, 2, 2, 2, 2, 2, 1]);
}

// ============================================================================
// 8. Stress Test: Non-Square Matrices Algebraic Invariants
// ============================================================================

#[test]
fn challenge_non_square_matrix_invariants() {
    // Transpose property: (A * B)^T == B^T * A^T for rectangular matrices
    // A: 5 x 2, B: 2 x 7 -> A * B: 5 x 7
    let a = Matrix::from_shape_fn(5, 2, |r, c| (r * 3 + c + 1) as f64).unwrap();
    let b = Matrix::from_shape_fn(2, 7, |r, c| (r * 2 + c + 2) as f64).unwrap();

    let ab = a.matmul(&b).unwrap();
    let ab_t = ab.transpose();

    let b_t = b.transpose();
    let a_t = a.transpose();
    let bt_at = b_t.matmul(&a_t).unwrap();

    assert_eq!(ab_t.rows, 7);
    assert_eq!(ab_t.cols, 5);
    assert_eq!(ab_t, bt_at);

    // Distributivity: A * (B + C) == A * B + A * C
    // A: 3 x 4, B: 4 x 2, C: 4 x 2
    let a_dist = Matrix::from_shape_fn(3, 4, |r, c| (r + c + 1) as f64).unwrap();
    let b_dist = Matrix::from_shape_fn(4, 2, |r, c| (r * 2 + c) as f64).unwrap();
    let c_dist = Matrix::from_shape_fn(4, 2, |r, c| (r + c * 3) as f64).unwrap();

    let b_plus_c = b_dist.add(&c_dist).unwrap();
    let left = a_dist.matmul(&b_plus_c).unwrap();

    let ab_dist = a_dist.matmul(&b_dist).unwrap();
    let ac_dist = a_dist.matmul(&c_dist).unwrap();
    let right = ab_dist.add(&ac_dist).unwrap();

    for (i, (&l, &r)) in left.data.iter().zip(&right.data).enumerate() {
        assert!((l - r).abs() < EPSILON, "Distributivity failed at {}: {} vs {}", i, l, r);
    }

    // Outer product vs Inner product
    // Column vector (4 x 1) and Row vector (1 x 4)
    let col_vec = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let row_vec = Matrix::from_vec(1, 4, vec![2.0, 3.0, 4.0, 5.0]).unwrap();

    // Inner product: (1 x 4) * (4 x 1) -> (1 x 1)
    let inner = row_vec.matmul(&col_vec).unwrap();
    assert_eq!(inner.rows, 1);
    assert_eq!(inner.cols, 1);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    assert_eq!(inner.data[0], 40.0);

    // Outer product: (4 x 1) * (1 x 4) -> (4 x 4)
    let outer = col_vec.matmul(&row_vec).unwrap();
    assert_eq!(outer.rows, 4);
    assert_eq!(outer.cols, 4);
    assert_eq!(outer[(0, 0)], 2.0);
    assert_eq!(outer[(3, 3)], 20.0);

    // Associativity of 3 rectangular matrices: (A * B) * C == A * (B * C)
    // A: 4 x 3, B: 3 x 5, C: 5 x 2
    let m_a = Matrix::from_shape_fn(4, 3, |r, c| (r + c + 1) as f64).unwrap();
    let m_b = Matrix::from_shape_fn(3, 5, |r, c| (r * 2 + c + 1) as f64).unwrap();
    let m_c = Matrix::from_shape_fn(5, 2, |r, c| (r + c * 3 + 1) as f64).unwrap();

    let ab_c = m_a.matmul(&m_b).unwrap().matmul(&m_c).unwrap();
    let a_bc = m_a.matmul(&m_b.matmul(&m_c).unwrap()).unwrap();

    assert_eq!(ab_c.rows, 4);
    assert_eq!(ab_c.cols, 2);
    assert_eq!(ab_c, a_bc);
}

// ============================================================================
// 9. Rejection of Zero Strides / Dimensions and Invalid Inputs
// ============================================================================

#[test]
fn challenge_zero_dimension_and_empty_rejection() {
    // Matrix
    assert!(Matrix::zeros(0, 0).is_err());
    assert!(Matrix::zeros(0, 5).is_err());
    assert!(Matrix::zeros(5, 0).is_err());
    assert!(Matrix::from_vec(0, 0, vec![]).is_err());
    assert!(Matrix::from_vec(0, 5, vec![]).is_err());
    assert!(Matrix::from_vec(5, 0, vec![]).is_err());
    assert!(Matrix::identity(0).is_err());

    // Buffer length mismatch
    assert!(Matrix::from_vec(3, 4, vec![0.0; 11]).is_err());
    assert!(Matrix::from_vec(3, 4, vec![0.0; 13]).is_err());

    // Tensor
    assert_eq!(Tensor::zeros(vec![]), Err(DspError::EmptyInput));
    assert_eq!(Tensor::from_vec(vec![], vec![]), Err(DspError::EmptyInput));

    // Zero dimensions in any axis
    assert!(Tensor::zeros(vec![0]).is_err());
    assert!(Tensor::zeros(vec![0, 5, 2]).is_err());
    assert!(Tensor::zeros(vec![2, 0, 3]).is_err());
    assert!(Tensor::zeros(vec![2, 3, 0]).is_err());

    assert!(Tensor::from_vec(vec![0], vec![]).is_err());
    assert!(Tensor::from_vec(vec![2, 0, 3], vec![]).is_err());

    // Tensor buffer length mismatch
    assert!(Tensor::from_vec(vec![2, 3, 4], vec![0.0; 23]).is_err());
    assert!(Tensor::from_vec(vec![2, 3, 4], vec![0.0; 25]).is_err());

    // Multi-index out of bounds
    let t = Tensor::zeros(vec![3, 4]).unwrap();
    assert!(t.get(&[3, 0]).is_err()); // index 3 out of bounds for dim 0
    assert!(t.get(&[0, 4]).is_err()); // index 4 out of bounds for dim 1
    assert!(t.get(&[0]).is_err());    // rank mismatch (1 != 2)
    assert!(t.get(&[0, 0, 0]).is_err()); // rank mismatch (3 != 2)

    // Submatrix out of bounds
    let m = Matrix::zeros(4, 4).unwrap();
    assert!(m.submatrix(2, 2, 0, 2).is_err()); // row_start == row_end
    assert!(m.submatrix(3, 2, 0, 2).is_err()); // row_start > row_end
    assert!(m.submatrix(0, 5, 0, 2).is_err()); // row_end > rows
    assert!(m.submatrix(0, 2, 2, 2).is_err()); // col_start == col_end
    assert!(m.submatrix(0, 2, 0, 5).is_err()); // col_end > cols
}

// ============================================================================
// 10. Verification of Constructor Aliases (`new`) and Data Accessors (`data`)
// ============================================================================

#[test]
fn challenge_constructor_aliases_and_data_accessors() {
    // 1. Matrix::new vs Matrix::from_vec equivalence
    let raw_mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m_new = Matrix::new(2, 3, raw_mat.clone()).expect("Matrix::new failed");
    let m_from_vec = Matrix::from_vec(2, 3, raw_mat.clone()).expect("Matrix::from_vec failed");
    assert_eq!(m_new, m_from_vec);
    assert_eq!(m_new.rows(), 2);
    assert_eq!(m_new.cols(), 3);
    assert_eq!(m_new.len(), 6);
    assert!(!m_new.is_empty());

    // 2. Matrix::data accessor equivalence
    assert_eq!(m_new.data(), raw_mat.as_slice());
    assert_eq!(m_new.data(), m_new.as_slice());
    assert_eq!(m_new.data(), &m_new.data[..]);

    // 3. Matrix::new adversarial edge cases and error mapping
    assert_eq!(Matrix::new(0, 0, vec![]), Err(DspError::EmptyInput));
    assert_eq!(Matrix::new(0, 4, vec![]), Err(DspError::EmptyInput));
    assert_eq!(Matrix::new(4, 0, vec![]), Err(DspError::EmptyInput));
    assert_eq!(
        Matrix::new(0, 4, vec![1.0, 2.0, 3.0, 4.0]),
        Err(DspError::invalid_parameter(
            "dimensions",
            "Matrix dimensions must be non-zero positive integers"
        ))
    );
    assert_eq!(
        Matrix::new(2, 3, vec![1.0; 5]),
        Err(DspError::buffer_length_mismatch(6, 5))
    );
    assert_eq!(
        Matrix::new(2, 3, vec![1.0; 7]),
        Err(DspError::buffer_length_mismatch(6, 7))
    );
    assert_eq!(
        Matrix::new(usize::MAX, 2, vec![]),
        Err(DspError::invalid_parameter("dimensions", "Matrix size overflowed usize"))
    );

    // 4. Tensor::new vs Tensor::from_vec equivalence
    let raw_tensor = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let t_new = Tensor::new(vec![2, 2, 2], raw_tensor.clone()).expect("Tensor::new failed");
    let t_from_vec = Tensor::from_vec(vec![2, 2, 2], raw_tensor.clone()).expect("Tensor::from_vec failed");
    assert_eq!(t_new, t_from_vec);
    assert_eq!(t_new.ndim(), 3);
    assert_eq!(t_new.shape(), &[2, 2, 2]);
    assert_eq!(t_new.len(), 8);
    assert!(!t_new.is_empty());

    // 5. Tensor::data and Tensor::data_mut accessors
    assert_eq!(t_new.data(), raw_tensor.as_slice());
    assert_eq!(t_new.data(), &t_new.data[..]);

    let mut t_mut = t_new.clone();
    t_mut.data_mut()[3] = 99.5;
    assert_eq!(t_mut.get(&[0, 1, 1]).unwrap(), 99.5);

    // 6. Tensor::new adversarial edge cases and error mapping
    assert_eq!(Tensor::new(vec![], vec![]), Err(DspError::EmptyInput));
    assert_eq!(
        Tensor::new(vec![0], vec![]),
        Err(DspError::invalid_parameter(
            "shape",
            "Tensor dimension sizes must be positive non-zero integers"
        ))
    );
    assert_eq!(
        Tensor::new(vec![2, 0, 3], vec![]),
        Err(DspError::invalid_parameter(
            "shape",
            "Tensor dimension sizes must be positive non-zero integers"
        ))
    );
    assert_eq!(
        Tensor::new(vec![2, 2, 2], vec![1.0; 7]),
        Err(DspError::buffer_length_mismatch(8, 7))
    );
    assert_eq!(
        Tensor::new(vec![2, 2, 2], vec![1.0; 9]),
        Err(DspError::buffer_length_mismatch(8, 9))
    );
    assert_eq!(
        Tensor::new(vec![usize::MAX, 2], vec![]),
        Err(DspError::invalid_parameter("shape", "Total tensor element count overflowed usize"))
    );
}

// ============================================================================
// 11. Allocation Safety with `checked_mul` on Adversarial Dimension Pairs
// ============================================================================

#[test]
fn challenge_checked_mul_matmul_allocation_safety() {
    // Test 1: Adversarial matrix dimensions where inner product matches, but outer allocation m * n overflows usize.
    // Matrix A is (usize::MAX x 1), Matrix B is (1 x 2).
    // A.cols == B.rows (1 == 1), but target allocation m * n = usize::MAX * 2 overflows.
    let mat_a_huge_rows = Matrix {
        rows: usize::MAX,
        cols: 1,
        data: vec![1.0],
    };
    let mat_b_col_2 = Matrix::new(1, 2, vec![2.0, 3.0]).unwrap();

    let matmul_res_1 = mat_a_huge_rows.matmul(&mat_b_col_2);
    match matmul_res_1 {
        Err(DspError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "dimensions");
            assert!(reason.contains("overflowed usize"), "Unexpected reason: {}", reason);
        }
        other => panic!("Expected InvalidParameter dimensions overflow, got {:?}", other),
    }

    // Test 2: Symmetric case: Matrix A is (2 x 1), Matrix B is (1 x usize::MAX).
    // Target allocation m * n = 2 * usize::MAX overflows.
    let mat_a_row_2 = Matrix::new(2, 1, vec![1.0, 2.0]).unwrap();
    let mat_b_huge_cols = Matrix {
        rows: 1,
        cols: usize::MAX,
        data: vec![1.0],
    };

    let matmul_res_2 = mat_a_row_2.matmul(&mat_b_huge_cols);
    match matmul_res_2 {
        Err(DspError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "dimensions");
            assert!(reason.contains("overflowed usize"), "Unexpected reason: {}", reason);
        }
        other => panic!("Expected InvalidParameter dimensions overflow, got {:?}", other),
    }

    // Test 3: Large but not MAX: (usize::MAX / 2 + 1) * 2
    let mat_a_half_max = Matrix {
        rows: usize::MAX / 2 + 1,
        cols: 1,
        data: vec![1.0],
    };
    let matmul_res_3 = mat_a_half_max.matmul(&mat_b_col_2);
    match matmul_res_3 {
        Err(DspError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "dimensions");
            assert!(reason.contains("overflowed usize"), "Unexpected reason: {}", reason);
        }
        other => panic!("Expected InvalidParameter dimensions overflow, got {:?}", other),
    }

    // Test 4: Matrix::zeros overflow protection
    assert_eq!(
        Matrix::zeros(usize::MAX, 2),
        Err(DspError::invalid_parameter("dimensions", "Matrix size overflowed usize"))
    );
    assert_eq!(
        Matrix::zeros(2, usize::MAX),
        Err(DspError::invalid_parameter("dimensions", "Matrix size overflowed usize"))
    );
    assert_eq!(
        Matrix::zeros(1 << 33, 1 << 33),
        Err(DspError::invalid_parameter("dimensions", "Matrix size overflowed usize"))
    );

    // Test 5: Tensor::zeros and Tensor::from_vec multi-dimension overflow protection
    assert_eq!(
        Tensor::zeros(vec![usize::MAX, 2]),
        Err(DspError::invalid_parameter("shape", "Total tensor element count overflowed usize"))
    );
    assert_eq!(
        Tensor::zeros(vec![2, usize::MAX / 2 + 1]),
        Err(DspError::invalid_parameter("shape", "Total tensor element count overflowed usize"))
    );
    assert_eq!(
        Tensor::zeros(vec![1 << 22, 1 << 22, 1 << 22]),
        Err(DspError::invalid_parameter("shape", "Total tensor element count overflowed usize"))
    );
    assert_eq!(
        Tensor::zeros(vec![2, 2, 2, 2, usize::MAX / 8]),
        Err(DspError::invalid_parameter("shape", "Total tensor element count overflowed usize"))
    );
}

// ============================================================================
// 12. Stress Testing Fold/Unfold Roundtrip across Prime and Extreme Topologies
// ============================================================================

#[test]
fn challenge_fold_unfold_prime_and_extreme_aspect_ratios() {
    let extreme_shapes: Vec<Vec<usize>> = vec![
        vec![100, 1],               // extreme aspect ratio tall
        vec![1, 100],               // extreme aspect ratio wide
        vec![1, 1, 64],             // singleton leading
        vec![64, 1, 1],             // singleton trailing
        vec![1, 64, 1],             // singleton middle
        vec![3, 5, 7],              // small primes
        vec![7, 11, 13],            // medium primes (1001 elements)
        vec![2, 3, 5, 7],           // 4D primes (210 elements)
        vec![2, 1, 3, 1, 5],        // 5D interspersed singletons (30 elements)
        vec![1, 1, 1, 1, 1],        // 5D all-ones singleton
    ];

    for shape in extreme_shapes {
        let ndim = shape.len();
        let tensor = Tensor::from_shape_fn(shape.clone(), |idx| {
            let mut h = 17_usize;
            for (i, &coord) in idx.iter().enumerate() {
                h = h.wrapping_mul(31).wrapping_add((coord + 1) * (i + 1));
            }
            (h % 1000) as f64 * 0.125
        })
        .unwrap();

        for mode in 0..ndim {
            let unfolded = tensor
                .mode_unfold(mode)
                .unwrap_or_else(|e| panic!("unfold failed on mode {} for shape {:?}: {:?}", mode, shape, e));

            assert_eq!(unfolded.rows, shape[mode]);
            let expected_cols: usize = shape.iter().enumerate().filter(|&(k, _)| k != mode).map(|(_, &d)| d).product();
            assert_eq!(unfolded.cols, expected_cols);

            let folded = Tensor::fold(&unfolded, mode, &shape)
                .unwrap_or_else(|e| panic!("fold failed on mode {} for shape {:?}: {:?}", mode, shape, e));

            assert_eq!(folded, tensor, "Lossless roundtrip failed for shape {:?}, mode {}", shape, mode);
        }
    }
}

// ============================================================================
// 13. Stress Testing Adversarial Mode Products with Forged Boundaries
// ============================================================================

#[test]
fn challenge_adversarial_mode_product_dimension_overflow() {
    let t = Tensor::from_shape_fn(vec![2, 3, 4], |idx| (idx[0] + idx[1] + idx[2]) as f64).unwrap();

    // 1. Forged matrix with rows = 0
    let fake_zero_rows = Matrix {
        rows: 0,
        cols: 3,
        data: vec![],
    };
    let res_zero = t.mode_product(1, &fake_zero_rows);
    assert!(res_zero.is_err(), "Zero rows forged matrix must return Err");

    // 2. Forged matrix with rows = usize::MAX
    let fake_overflow_rows = Matrix {
        rows: usize::MAX,
        cols: 3,
        data: vec![],
    };
    let res_overflow = t.mode_product(1, &fake_overflow_rows);
    match res_overflow {
        Err(DspError::InvalidParameter { name, reason }) => {
            assert_eq!(name, "dimensions");
            assert!(reason.contains("overflowed usize"));
        }
        other => panic!("Expected dimension overflow error, got {:?}", other),
    }

    // 3. Mode product on 1D tensor
    let t1 = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let mat_1d = Matrix::from_shape_fn(2, 4, |r, c| (r + c + 1) as f64).unwrap();
    let res_1d = t1.mode_product(0, &mat_1d).unwrap();
    assert_eq!(res_1d.shape(), &[2]);
    // y[0] = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
    // y[1] = 2*1 + 3*2 + 4*3 + 5*4 = 2 + 6 + 12 + 20 = 40
    assert_eq!(res_1d.data(), &[30.0, 40.0]);
}


