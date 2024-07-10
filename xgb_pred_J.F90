
!EOC
!------------------------------------------------------------------------------
!                  GEOS-Chem Global Chemical Transport Model                  !
!------------------------------------------------------------------------------
!BOP
!
! !ROUTINE: xgb_pred_J
!
! !DESCRIPTION: Subroutine ! Predict phot rates
!
!
!\\
!\\
! !INTERFACE:
!
  SUBROUTINE xgb_pred_J(State_Chm, State_Grid, State_Met)

    USE State_Chm_Mod,   ONLY : ChmState
    USE State_Met_Mod,   ONLY : MetState
    USE State_Grid_Mod,  ONLY : GrdState
    USE CMN_FJX_Mod,     ONLY : NRATJ, L_
    USE TIME_MOD,        ONLY : GET_MONTH, GET_DAY, GET_DAY_OF_YEAR
    USE TIME_MOD,        ONLY : GET_TAU,   GET_YEAR
    USE TOMS_MOD,        ONLY : GET_OVERHEAD_O3
    USE CMN_SIZE_MOD,    ONLY : NDUST
    USE Grid_Registry_Mod
    USE Pressure_Mod
    USE xgb_fortran_api
    USE iso_c_binding
    IMPLICIT NONE

    TYPE(ChmState), INTENT(IN)  :: State_Chm
    TYPE(GrdState), INTENT(IN)  :: State_Grid
    TYPE(MetState), INTENT(IN)  :: State_Met


    ! FOR INIT
    LOGICAL, SAVE                 :: first_time = .TRUE.
    REAL(c_float), ALLOCATABLE    :: xx_carr_small(:,:)
    INTEGER(c_int64_t)            :: xx_dmtrx_len, nrow_dummy
    CHARACTER(LEN=255)            :: xx_fname

    ! LOCAL VARS
    INTEGER(c_int64_t)            :: xx_param_count
    INTEGER(c_int)                :: xx_option_mask, xx_ntree_limit, xx_training
    INTEGER(c_int)                :: xx_rc
    REAL(c_float), parameter      :: missing_value = -999.0
    TYPE(c_ptr)                   :: xx_dmtrx
    TYPE(c_ptr), SAVE             :: xx_booster
    INTEGER(c_int64_t)            :: xx_prediction_count, xx_count ! How many grid boxes: NZ * NX * NY
    REAL(c_float), ALLOCATABLE    :: xx_carr(:,:)
    REAL(fp) :: xx_u0, xx_sza, xx_solf
    INTEGER  :: xx_prediction_index, DAY_OF_YR
    REAL(fp), POINTER :: ODMDUST  (:,:,:,:,:)

    ! FOR PREDICTION

    INTEGER(c_int64_t)            :: xx_pred_len
    TYPE(c_ptr)                   :: xx_cpred

    REAL(c_float), POINTER     :: xx_pred(:)

    ! TEMPORARY LOCAL VARS
    INTEGER                          :: xx_lon, xx_lat, xx_lev, i, J, xx_index, i_sum, largest_lon, largest_lat, largest_lev
    INTEGER                          :: xx_n
    REAL, ALLOCATABLE                :: J_ML(:,:,:,:)
    REAL                             :: P0, HyAm, HyBm, Lev, XMID, YMID, POS_ENC, start, finish, largest_value
    REAL                             :: TAUCLW_sum_above, TAUCLW_sum_below, TAUCLI_sum_above, TAUCLI_sum_below, CLDF_sum_above, CLDF_sum_below

    REAL(fp), POINTER                :: ZPJ      (:,:,:,:)


    integer(c_int64_t) :: xx_params, xx_total_preds, xx_nrows, xx_ncols

    ! FOR FIRST TIME INIT
    TYPE(c_ptr), DIMENSION(166), SAVE :: xx_boosters

    TYPE ModelInfo
        INTEGER :: modelID
        CHARACTER(LEN=255) :: filePath
        REAL(c_float) :: constant
        CHARACTER(LEN=255) :: species
        CHARACTER(LEN=255) :: predictor
        REAL(c_float)      :: factor
    END TYPE ModelInfo
    TYPE(ModelInfo), DIMENSION(166) :: models


    models(1) = ModelInfo(1, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_O2.model', 8.346e-41, 'O2', 'Unique', 1.000)
    models(2) = ModelInfo(2, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_JvalO3O3P.model', 1.1831617e-09, 'JvalO3O3P', 'Unique', 1.000)
    models(3) = ModelInfo(3, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_JvalO3O1D.model', 1.2237375e-11, 'JvalO3O1D', 'Unique', 1.000)
    !Models 4 and 5 have not a value to be trained off so use a default model and 0.0 multiplier
    models(4) = ModelInfo(4, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'temporary', 'MISSING', 0.000)
    models(5) = ModelInfo(5, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'temporary', 'MISSING', 0.000)
    models(6) = ModelInfo(6, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_NO.model', 1.021373e-39, 'NO', 'Unique', 1.000)
    models(7) = ModelInfo(7, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H2COa.model', 4.2749283e-15, 'CH2Oa', 'Unique', 1.000)
    models(8) = ModelInfo(8, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H2COb.model', 1.845236e-14, 'CH2Ob', 'Unique', 1.000)
    models(9) = ModelInfo(9, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H2O2.model', 1.9968258e-11, 'H2O2', 'Unique', 1.000)
    models(10) = ModelInfo(10, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'MP', 'CH3OOH', 1.0)
    models(11) = ModelInfo(11, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_NO2.model', 6.090331e-08, 'NO2', 'Unique', 1.000)
    models(12) = ModelInfo(12, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_NO3.model', 5.673017e-07, 'NO3', 'Unique', 0.886)
    models(13) = ModelInfo(13, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_NO3.model', 5.673017e-07, 'NO3', 'Unique', 0.114)
    models(14) = ModelInfo(14, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_N2O5.model', 1.5224057e-10, 'N2O5', 'Unique', 1.000)
    models(15) = ModelInfo(15, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HNO2.model', 9.547056e-09, 'HNO2', 'Unique', 1.000)
    models(16) = ModelInfo(16, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HNO3.model', 8.272876e-13, 'HNO3', 'Unique', 1.000)
    models(17) = ModelInfo(17, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HNO4.model', 4.6425617e-11, 'HNO4', 'Unique', 0.05)
    models(18) = ModelInfo(18, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HNO4.model', 4.6425617e-11, 'HNO4', 'Unique', 0.95)
    models(19) = ModelInfo(19, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ClNO3a.model', 1.3835245e-13, 'ClNO3a', 'Unique', 1.000)
    models(20) = ModelInfo(20, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ClNO3b.model', 4.2134735e-15, 'ClNO3b', 'Unique', 1.000)
    models(21) = ModelInfo(21, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ClNO2.model', 1.481781e-09, 'ClNO2', 'Unique', 1.000)
    models(22) = ModelInfo(22, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Cl2.model', 1.209261e-08, 'Cl2', 'Unique', 1.000)
    models(23) = ModelInfo(23, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Br2.model', 1.1601841e-07, 'Br2', 'Unique', 1.000)
    models(24) = ModelInfo(24, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HOCl.model', 1.2624998e-09, 'HOCl', 'Unique', 1.000)
    models(25) = ModelInfo(25, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_OClO.model', 4.936651e-07, 'OClO', 'Unique', 1.000)
    models(26) = ModelInfo(26, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Cl2O2.model', 8.263315e-09, 'Cl2O2', 'Unique', 1.000)
    models(27) = ModelInfo(27, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ClO.model', 6.222837e-11, 'ClO', 'Unique', 1.000)
    models(28) = ModelInfo(28, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_BrO.model', 1.7312884e-07, 'BrO', 'Unique', 1.000)
    models(29) = ModelInfo(29, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_BrNO3.model', 6.898928e-09, 'BrNO3', 'Unique', 0.85)
    models(30) = ModelInfo(30, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_BrNO3.model', 6.898928e-09, 'BrNO3', 'Unique', 0.15)
    models(31) = ModelInfo(31, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_BrNO2.model', 3.0139745e-08, 'BrNO2', 'Unique', 1.000)
    models(32) = ModelInfo(32, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HOBr.model', 1.128905e-08, 'HOBr', 'Unique', 1.000)
    models(33) = ModelInfo(33, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_BrCl.model', 5.346989e-08, 'BrCl', 'Unique', 1.000)
    models(34) = ModelInfo(34, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_OCS.model', 3.118778e-21, 'OCS', 'Unique', 1.000)
    ! There should be model for SO2 but the jvals are nan so uses defualt model and 0.0 multiplier
    models(35) = ModelInfo(35, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'temporary', 'MISSING', 0.000)
    models(36) = ModelInfo(36, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_N2O.model', 1.1980623e-37, 'N2O', 'Unique', 1.000)
    models(37) = ModelInfo(37, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CFC11.model', 2.2066945e-36, 'CFC11', 'Unique', 1.000)
    models(38) = ModelInfo(38, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CFC12.model', 1.4267726e-37, 'CFC12', 'Unique', 1.000)
    models(39) = ModelInfo(39, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CFC113.model', 2.9244467e-37, 'CFC113', 'Unique', 1.000)
    models(40) = ModelInfo(40, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CFC114.model', 1.7984768e-38, 'CFC114', 'Unique', 1.000)
    models(41) = ModelInfo(41, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CFC115.model', 1.21646e-39, 'CFC115', 'Unique', 1.000)
    models(42) = ModelInfo(42, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CCl4.model', 6.072607e-36, 'CCl4', 'Unique', 1.000)
    models(43) = ModelInfo(43, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH3Cl.model', 3.497752e-38, 'CH3Cl', 'Unique', 1.000)
    models(44) = ModelInfo(44, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH3CCl3.model', 3.7910016e-36, 'CH3CCl3', 'Unique', 1.000)
    models(45) = ModelInfo(45, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH2Cl2.model', 2.341617e-37, 'CH2Cl2', 'Unique', 1.000)
    models(46) = ModelInfo(46, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HCFC22.model', 7.03871e-40, 'HCFC22', 'Unique', 1.000)
    models(47) = ModelInfo(47, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HCFC123.model', 2.9430889e-37, 'HCFC123', 'Unique', 1.000)
    models(48) = ModelInfo(48, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HCFC141b.model', 3.6372545e-37, 'HCFC141b', 'Unique', 1.000)
    models(49) = ModelInfo(49, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HCFC142b.model', 2.848345e-39, 'HCFC142b', 'Unique', 1.000)
    models(50) = ModelInfo(50, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH3Br.model', 1.155432e-25, 'CH3Br', 'Unique', 1.000)
    models(51) = ModelInfo(51, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H1211.model', 9.661064e-16, 'H1211', 'Unique', 1.000)
    models(52) = ModelInfo(52, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H1211.model', 9.661064e-16, 'H1210', 'H1211', 1.000)
    models(53) = ModelInfo(53, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H1301.model', 1.1605463e-20, 'H1301', 'Unique', 1.000)
    models(54) = ModelInfo(54, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_H2402.model', 3.0399666e-16, 'H2402', 'Unique', 1.000)
    models(55) = ModelInfo(55, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH2Br2.model', 6.613508e-18, 'CH2Br2', 'Unique', 1.000)
    models(56) = ModelInfo(56, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CHBr3.model', 1.3854249e-12, 'CHBr3', 'Unique', 1.000)
    !There does not exist a J=57 so uses defualt model and 0.0 multiplier
    models(57) = ModelInfo(57, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'NON EXISTANT', 'NON EXISTANT', 0.0)
    models(58) = ModelInfo(58, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CF3I.model', 1.1302509e-14, 'CF3I', 'MISSING', 1.000)
    models(59) = ModelInfo(59, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_PAN.model', 1.5455034e-12, 'PAN', 'Unique', 1.000)
    models(60) = ModelInfo(60, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_R4N2.model', 2.2864553e-12, 'R4N2', 'Unique', 0.5)
    models(61) = ModelInfo(61, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ActAld.model', 8.7724555e-16, 'ALD2 (ActAld)', 'Unique', 1.000)
    !Jval returns nan so uses defualt model and 0.0 multiplier
    models(62) = ModelInfo(62, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'ALD2 (ActAlX)', 'returns nan', 0.000)
    models(63) = ModelInfo(63, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MVK.model', 1.41704435e-11, 'MVK', 'Unique', 0.6)
    models(64) = ModelInfo(64, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MVK.model', 1.41704435e-11, 'MVK', 'Unique', 0.2)
    models(65) = ModelInfo(65, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MVK.model', 1.41704435e-11, 'MVK', 'Unique', 0.2)
    models(66) = ModelInfo(66, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MACR.model', 7.636825e-12, 'MACR', 'Unique', 1.0)
    models(67) = ModelInfo(67, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MACR.model', 7.636825e-12, 'MACR', 'Unique', 0.0)
    models(68) = ModelInfo(68, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_GLYC.model', 1.31978395e-11, 'GLYC', 'Unique', 1.000)
    models(69) = ModelInfo(69, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MEK.model', 4.9985376e-12, 'MEK', 'Unique', 1.000)
    models(70) = ModelInfo(70, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_RCHO.model', 9.514112e-11, 'RCHO', 'Unique', 1.000)
    models(71) = ModelInfo(71, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MGLY.model', 8.9409086e-10, 'MGLY', 'Unique', 1.000)
    models(72) = ModelInfo(72, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Glyxla.model', 1.687191e-13, 'GLYXA', 'Unique', 1.000)
    models(73) = ModelInfo(73, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Glyxlb.model', 9.007525e-15, 'GLYXB', 'Unique', 1.000)
    models(74) = ModelInfo(74, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Glyxlc.model', 3.837299e-14, 'GLYXC', 'Unique', 1.000)
    models(75) = ModelInfo(75, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HAC.model', 4.1262545e-12, 'HAC', 'Unique', 1.000)
    models(76) = ModelInfo(76, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Acet-a.model', 7.532465e-17, 'ACETa', 'Unique', 1.000)
    models(77) = ModelInfo(77, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_Acet-b.model',  4.073666e-18, 'ACETb', 'Unique', 1.000)
    models(78) = ModelInfo(78, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IDN.model', 1.0134167e-11, 'IDN', 'Unique', 1.000)
    models(79) = ModelInfo(79, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'PRPN', 'CH3OOH', 1.0)
    models(80) = ModelInfo(80, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'ETP', 'CH3OOH', 0.5)
    models(81) = ModelInfo(81, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RA3P', 'CH3OOH', 1.0)
    models(82) = ModelInfo(82, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RB3P', 'CH3OOH', 1.0)
    models(83) = ModelInfo(83, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'R4P', 'CH3OOH', 1.0)
    models(84) = ModelInfo(84, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'PP', 'CH3OOH', 1.0)
    models(85) = ModelInfo(85, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RP', 'CH3OOH', 1.0)
    models(86) = ModelInfo(86, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HMHP.model', 1.3574333e-11, 'HMHP', 'Unique', 1.000)
    models(87) = ModelInfo(87, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'HPETHNL', 'PrAldP', 1.000)
    models(88) = ModelInfo(88, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_PYAC.model', 8.9409086e-10, 'PYAC', 'Unique', 1.000)
    models(89) = ModelInfo(89, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_PROPNN.model', 6.7912974e-11, 'PROPNN', 'Unique', 1.000)
    models(90) = ModelInfo(90, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MVKHC.model', 8.9409086e-10, 'MVKHC', 'Unique', 1.000)
    models(91) = ModelInfo(91, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_RCHO.model', 9.514112e-11, 'MVKHCB', 'RCHO', 1.000)
    models(92) = ModelInfo(92, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'MVKHP', 'CH3OOH', 1.0)
    models(93) = ModelInfo(93, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'MVKPC', 'PrAldP', 1.000)
    models(94) = ModelInfo(94, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MCRENOL.model', 1.4701507e-09, 'MCRENOL', 'Unique', 1.000)
    models(95) = ModelInfo(95, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'MCRHP', 'PrAldP', 1.000)
    models(96) = ModelInfo(96, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'MACR1OOH', 'PrAldP', 1.000)
    models(97) = ModelInfo(97, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'ATOOH', 'CH3OOH', 1.0)
    models(98) = ModelInfo(98, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_R4N2.model', 2.2864553e-12, 'R4N2', 'Unique', 0.5)
    models(99) = ModelInfo(99, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'MAP', 'CH3OOH', 1.0)
    models(100) = ModelInfo(100, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_SO4.model', 3.0301065e-18, 'SO4', 'Unique', 1.000)
    models(101) = ModelInfo(101, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ClOO.model', 1.5329666e-06, 'ClOO', 'Unique', 1.000)
    models(102) = ModelInfo(102, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_O3.model', 1.2237375e-11, 'O3', 'Unique', 1.000)
    models(103) = ModelInfo(103, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MPN.model', 4.3990675e-12, 'MPN', 'Unique', 0.05)
    models(104) = ModelInfo(104, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MPN.model', 4.3990675e-12, 'MPN', 'Unique', 0.95)
    models(105) = ModelInfo(105, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_PIP.model', 1.9968258e-11, 'PIP', 'Unique', 1.000)
    models(106) = ModelInfo(106, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ICN.model', 1.4051836e-09, 'ICN', 'Unique', 1.000)
    models(107) = ModelInfo(107, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ETHLN.model', 3.4546888e-10, 'ETHLN', 'Unique', 1.000)
    models(108) = ModelInfo(108, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MVKN.model', 8.019085e-11, 'MVKN', 'Unique', 1.000)
    models(109) = ModelInfo(109, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MCRHN.model', 9.64955e-10, 'MCRHN', 'Unique', 1.000)
    models(110) = ModelInfo(110, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MCRHNB.model', 2.4614208e-10, 'MCRHNB', 'Unique', 1.000)
    models(111) = ModelInfo(111, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'MONITS', 'ONIT1', 1.000)
    models(112) = ModelInfo(112, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'MONITU', 'ONIT1', 1.000)
    models(113) = ModelInfo(113, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'HONIT', 'ONIT1', 1.000)
    models(114) = ModelInfo(114, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_I2.model', 5.229235e-07, 'I2', 'Unique', 1.000)
    models(115) = ModelInfo(115, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HOI.model', 4.6213064e-08, 'HOI', 'Unique', 1.000)
    models(116) = ModelInfo(116, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IO.model', 9.4386473e-07, 'IO', 'Unique', 1.000)
    models(117) = ModelInfo(117, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_OIO.model', 6.0653554e-07, 'OIO', 'Unique', 1.000)
    models(118) = ModelInfo(118, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_INO.model', 1.4956237e-07, 'INO', 'Unique', 1.000)
    models(119) = ModelInfo(119, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IONO.model', 1.5672333e-08, 'IONO', 'Unique', 1.000)
    models(120) = ModelInfo(120, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IONO2.model', 5.517457e-08, 'IONO2', 'Unique', 1.000)
    models(121) = ModelInfo(121, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_I2O2.model', 1.9553251e-07, 'I2O2', 'Unique', 1.000)
    models(122) = ModelInfo(122, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH3I.model', 1.178823e-11, 'CH3I', 'Unique', 1.000)
    models(123) = ModelInfo(123, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH2I2.model', 2.7680464e-08, 'CH2I2', 'Unique', 1.000)
    models(124) = ModelInfo(124, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH2ICl.model', 3.770074e-10, 'CH2ICl', 'Unique', 1.000)
    models(125) = ModelInfo(125, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_CH2IBr.model', 1.395895e-09, 'CH2IBr', 'Unique', 1.000)
    models(126) = ModelInfo(126, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_I2O4.model', 1.9553251e-07, 'I2O4', 'Unique', 1.000)
    models(127) = ModelInfo(127, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_I2O3.model', 1.937503e-07, 'I2O3', 'Unique', 1.000)
    models(128) = ModelInfo(128, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IBr.model', 2.2424717e-07, 'IBr', 'Unique', 1.000)
    models(129) = ModelInfo(129, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ICl.model', 7.884039e-08, 'ICl', 'Unique', 1.000)
    !130->133 returns illogical values uses 0.0 multiplier
    models(130) = ModelInfo(130, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'NITs', 'Unique', 0.0)
    models(131) = ModelInfo(131, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'NITs', 'Unique', 0.0)
    models(132) = ModelInfo(132, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'NIT', 'Unique', 0.0)
    models(133) = ModelInfo(133, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'NIT', 'Unique', 0.0)
    models(134) = ModelInfo(134, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MENO3.model', 1.1317352e-12, 'MENO3', 'Unique', 1.000)
    models(135) = ModelInfo(135, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_ETNO3.model', 2.057413e-12, 'ETNO3', 'Unique', 1.000)
    models(136) = ModelInfo(136, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IPRNO3.model', 3.787802e-12, 'IPRNO3', 'Unique', 1.000)
    models(137) = ModelInfo(137, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_NPRNO3.model', 3.4222401e-12, 'NPRNO3', 'Unique', 1.000)
    models(138) = ModelInfo(138, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RIPA', 'CH3OOH', 1.0)
    models(139) = ModelInfo(139, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RIPB', 'CH3OOH', 1.0)
    models(140) = ModelInfo(140, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RIPC', 'CH3OOH', 1.0)
    models(141) = ModelInfo(141, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'RIPD', 'CH3OOH', 1.0)
    models(142) = ModelInfo(142, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPALD1.model', 1.4764497e-09, 'HPALD1', 'Unique', 1.000)
    models(143) = ModelInfo(143, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPALD2.model', 1.4001184e-09, 'HPALD2', 'Unique', 1.000)
    models(144) = ModelInfo(144, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'HPALD3', 'PrAldP', 1.000)
    models(145) = ModelInfo(145, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'HPALD4', 'PrAldP', 1.000)
    models(146) = ModelInfo(146, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'IHN1', 'ONIT1', 1.000)
    models(147) = ModelInfo(147, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'IHN2', 'ONIT1', 1.000)
    models(148) = ModelInfo(148, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'IHN3', 'ONIT1', 1.000)
    models(149) = ModelInfo(149, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'IHN4', 'ONIT1', 1.000)
    models(150) = ModelInfo(150, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_INPB.model', 2.445911e-11, 'INPB', 'Unique', 1.000)
    models(151) = ModelInfo(151, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'INPD', 'CH3OOH', 1.0)
    models(152) = ModelInfo(152, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'INPD', 'ONIT1', 1.000)
    models(153) = ModelInfo(153, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_RCHO.model', 9.514112e-11, 'ICPDH', 'RCHO', 1.000)
    models(154) = ModelInfo(154, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'ICPDH', 'CH3OOH', 1.0)
    models(155) = ModelInfo(155, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_IDHDP.model', 3.878883e-11, 'IDHDP', 'Unique', 1.000)
    models(156) = ModelInfo(156, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'IDHPE', 'CH3OOH', 1.0)
    models(157) = ModelInfo(157, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_HPETHNL.model', 1.14643794e-10, 'IDCHP', 'PrAldP', 1.000)
    models(158) = ModelInfo(158, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'ITHN', 'CH3OOH', 1.0)
    models(159) = ModelInfo(159, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MONITS.model', 5.0670835e-12, 'ITHN', 'ONIT1', 1.000)
    models(160) = ModelInfo(160, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MCRHNB.model', 2.4614208e-10, 'ITCN', 'MCRHNB', 1.000)
    models(161) = ModelInfo(161, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_RCHO.model', 9.514112e-11, 'ITCN', 'RCHO', 1.000)
    models(162) = ModelInfo(162, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'ETHP', 'CH3OOH', 1.0)
    models(163) = ModelInfo(163, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_BALD.model', 7.5236505e-11, 'BALD', 'Unique', 1.000)
    models(164) = ModelInfo(164, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'BZCO3H', 'CH3OOH', 1.0)
    models(165) = ModelInfo(165, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_MP.model', 1.93920279140114e-11, 'BENZP', 'CH3OOH', 1.0)
    models(166) = ModelInfo(166, '/users/vxv505/scratch/conda_environments/final_model/xgb_model_Jval_NPHEN.model', 6.7912974e-11, 'NPHEN', 'Unique', 1.000)



    ZPJ       => State_Chm%Phot%ZPJ
    ODMDUST   => State_Chm%Phot%ODMDUST
    xx_param_count = 19

    IF (ALLOCATED (J_ML)) DEALLOCATE(J_ML)
    IF (ALLOCATED (xx_carr)) DEALLOCATE(xx_carr)

    xx_count = 0
    DO xx_lon = 1, State_Grid%NX
    DO xx_lat = 1, State_Grid%NY
    DO xx_lev = 1, State_Grid%NZ
        DAY_OF_YR = GET_DAY_OF_YEAR()
        xx_u0 = State_Met%SUNCOSmid(xx_lon, xx_lat)
        CALL SOLAR_JX(DAY_OF_YR, xx_u0, xx_sza, xx_solf)

        IF (xx_sza < 98) THEN
                xx_count = xx_count + 1
        END IF
    END DO
    END DO
    END DO

    WRITE(6,*)'xx_count: ',xx_count
    ALLOCATE(xx_carr(xx_param_count, xx_count))
    xx_index = 1
    DO xx_lon = 1, State_Grid%NX
        DO xx_lat = 1, State_Grid%NY
            DO xx_lev = 1, State_Grid%NZ
                P0       = 1000.0_f8
                HyAm = ( Get_Ap( xx_lev ) + Get_Ap( xx_lev+1 ) ) * 0.5_f8
                HyBm = ( Get_Bp( xx_lev ) + Get_Bp(xx_lev+1 ) ) * 0.5_f8
                Lev = (HyAm/P0) +HyBm
                !WRITE(6,*)'TEST LEV ',Lev
                DAY_OF_YR = GET_DAY_OF_YEAR()
                xx_u0 = State_Met%SUNCOSmid(xx_lon, xx_lat)
                CALL SOLAR_JX(DAY_OF_YR, xx_u0, xx_sza, xx_solf)

                TAUCLI_sum_above = 0.0
                TAUCLI_sum_below = 0.0
                TAUCLW_sum_above = 0.0
                TAUCLW_sum_below = 0.0
                CLDF_sum_above = 0.0
                CLDF_sum_below = 0.0

                ! Sum for levels below current
                DO i_sum = 1, xx_lev - 1
                    TAUCLI_sum_below = TAUCLI_sum_below + State_Met%TAUCLI(xx_lon, xx_lat, i_sum)
                    TAUCLW_sum_below = TAUCLW_sum_below + State_Met%TAUCLW(xx_lon, xx_lat, i_sum)
                    CLDF_sum_below = CLDF_sum_below + State_Met%CLDF(xx_lon, xx_lat, i_sum)
                END DO

                ! Sum for levels above
                DO i_sum = xx_lev + 1, State_Grid%NZ
                    TAUCLI_sum_above = TAUCLI_sum_above + State_Met%TAUCLI(xx_lon, xx_lat, i_sum)
                    TAUCLW_sum_above = TAUCLW_sum_above + State_Met%TAUCLW(xx_lon, xx_lat, i_sum)
                    CLDF_sum_above = CLDF_sum_above + State_Met%CLDF(xx_lon, xx_lat, i_sum)
                END DO





                IF (xx_sza < 98) THEN



                    xx_carr(1, xx_index) = Lev
                    xx_carr(2, xx_index) = State_Met%SUNCOSmid(xx_lon, xx_lat)
                    xx_carr(3, xx_index) = State_Met%UVALBEDO(xx_lon, xx_lat)
                    xx_carr(4, xx_index) = GET_OVERHEAD_O3(State_Chm, xx_lon, xx_lat)
                    xx_carr(5, xx_index) = xx_sza
                    xx_carr(6, xx_index) = State_Met%PMid(xx_lon, xx_lat, xx_lev)
                    xx_carr(7, xx_index) = State_Met%T(xx_lon, xx_lat, xx_lev)
                    xx_carr(8, xx_index) = TAUCLW_sum_above
                    xx_carr(9, xx_index) = TAUCLW_sum_below
                    xx_carr(10, xx_index) = TAUCLI_sum_above
                    xx_carr(11, xx_index) = TAUCLI_sum_below
                    xx_carr(12, xx_index) = CLDF_sum_above
                    xx_carr(13, xx_index) = CLDF_sum_below
                    xx_carr(14, xx_index) = State_Met%AIRDEN(xx_lon, xx_lat, xx_lev)
                    xx_carr(15, xx_index) = State_Met%CLDF(xx_lon, xx_lat, xx_lev)
                    xx_carr(16, xx_index) = State_Met%TAUCLI(xx_lon, xx_lat, xx_lev)
                    xx_carr(17, xx_index) = State_Met%TAUCLW(xx_lon, xx_lat, xx_lev)
                    xx_carr(18, xx_index) =  ODMDUST(xx_lon, xx_lat, xx_lev, State_Chm%Phot%IWV1000, 1)
                    xx_carr(19, xx_index) =  ODMDUST(xx_lon, xx_lat, xx_lev, State_Chm%Phot%IWV1000, 7)

                    !IF (xx_lon .EQ. 65 .AND. xx_lat .EQ. 5 .AND. xx_lev .EQ. 68) THEN
                    !    DO i = 1, xx_param_count
                    !            WRITE(6,*) 'xx_carr(',i,', index)', xx_carr(i, xx_index)
                    !    END DO
                    !END IF
                    xx_index = xx_index + 1
                END IF
            END DO
        END DO
    END DO

    ALLOCATE(J_ML(State_Grid%NX, State_Grid%NY, State_Grid%NZ, 166))
    J_ML = 0.0

    xx_rc = XGDMatrixCreateFromMat_f(xx_carr, xx_count, xx_param_count, missing_value, xx_dmtrx)
    IF (xx_rc/= 0) THEN
        WRITE(6,*) 'Error in creating DMatrix in Run', xx_rc
    END IF

    IF (ALLOCATED (xx_carr)) DEALLOCATE(xx_carr)



    IF (first_time) THEN
        first_time = .FALSE.
        DO J = 1, 166
            xx_fname = TRIM(models(J)%filePath)

            xx_rc = XGBoosterCreate_f(c_null_ptr, xx_dmtrx_len, xx_boosters(J))
            IF (xx_rc/= 0) THEN
                    WRITE(6,*) 'Error in Creating Booster in Initialising', xx_rc
            END IF


            ! 3. Load Booster from File
            !WRITE(6,*) 'Reading File for XGBoost: ', xx_fname
            xx_rc = XGBoosterLoadModel_f(xx_boosters(J), xx_fname)
            WRITE(6,*) 'Initialised Model: ', models(J)%species

            IF (xx_rc/= 0) THEN
                    WRITE(6,*) 'Error in Loading Model in Initialising', xx_rc
                    WRITE(6,*) 'Error at: ', xx_fname
                    WRITE(6,*) 'Error for species: ', models(J)%species
            END IF
        END DO
    END IF


    CALL cpu_time(start)
    !$OMP PARALLEL DO SHARED(J_ML, xx_dmtrx, models, State_Grid, State_Met, ZPJ, xx_boosters) &  
    !$OMP PRIVATE(J, xx_lon, xx_lat, xx_lev, xx_index, DAY_OF_YR, xx_u0, xx_sza, xx_solf) & 
    !$OMP PRIVATE(xx_fname, xx_param_count, xx_option_mask, xx_ntree_limit, xx_training) &
    !$OMP PRIVATE(xx_dmtrx_len, xx_pred, xx_cpred, xx_booster, xx_rc) &  
    !$OMP SCHEDULE(DYNAMIC)

    DO J = 1, 166

        xx_option_mask = 0
        xx_ntree_limit = 0
        xx_training = 0
        xx_dmtrx_len = 0

        xx_rc = XGBoosterPredict_f(xx_boosters(J), xx_dmtrx, xx_option_mask, xx_ntree_limit, xx_training, xx_pred_len, xx_cpred)
        IF (xx_rc/= 0) THEN
                WRITE(6,*) 'Error in XGBooster Predicting in Run', xx_rc
        END IF

        IF (ASSOCIATED(xx_pred)) NULLIFY(xx_pred)

        !write(6,*) 'PREDICTION LENGHT: ', xx_pred_len
        call c_f_pointer(xx_cpred, xx_pred, [xx_pred_len])

        IF (.NOT. ASSOCIATED(xx_pred)) THEN
            WRITE(6,*) 'Error: xx_pred is not associated.'
        END IF

        xx_index = 1
        DO xx_lon = 1, State_Grid%NX
        DO xx_lat = 1, State_Grid%NY
        DO xx_lev = 1, State_Grid%NZ
            DAY_OF_YR = GET_DAY_OF_YEAR()
            xx_u0 = State_Met%SUNCOSmid(xx_lon, xx_lat)
            CALL SOLAR_JX(DAY_OF_YR, xx_u0, xx_sza, xx_solf)

            IF (xx_sza < 98) THEN

                J_ML(xx_lon, xx_lat, xx_lev, J) = (EXP(xx_pred(xx_index)) - models(J)%constant) * models(J)%factor

                xx_index=xx_index+1

            END IF
        END DO
        END DO
        END DO

        IF (ASSOCIATED(xx_pred)) NULLIFY(xx_pred)
        !IF (ASSOCIATED(xx_cpred)) NULLIFY(xx_cpred)
        xx_cpred = c_null_ptr

        !xx_rc = XGBoosterFree_f(xx_booster)



        DO xx_lon = 1, State_Grid%NX
        DO xx_lat = 1, State_Grid%NY
        DO xx_lev = 1, State_Grid%NZ
                ZPJ(xx_lev, J, xx_lon, xx_lat) = J_ML(xx_lon, xx_lat, xx_lev, J)
        END DO
        END DO
        END DO

    END DO

    !$OMP END PARALLEL DO 
    DO xx_lon = 1, State_Grid%NX
    DO xx_lat = 1, State_Grid%NY
    DO xx_lev = 1, State_Grid%NZ
    DO J = 1, 166
        IF (ZPJ(xx_lev, J, xx_lon, xx_lat) < 0.0) THEN
                ZPJ(xx_lev, J, xx_lon, xx_lat) = 0.0
        END IF
    END DO
    END DO
    END DO
    END DO


    CALL cpu_time(finish)
    WRITE(6,*)'Time to Predict: ',finish-start
    xx_rc = XGDMatrixFree_f(xx_dmtrx)
    IF (ALLOCATED (J_ML)) DEALLOCATE(J_ML)

    ODMDUST   => NULL()
    ZPJ       => NULL()

  END SUBROUTINE xgb_pred_J


