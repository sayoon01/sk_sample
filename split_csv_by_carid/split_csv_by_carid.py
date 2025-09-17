# -*- coding:utf-8 -*-

import os
import pandas as pd
import sys
from collections import defaultdict
import fileinput
import numpy as np
import time
import datetime
import ray
import traceback


#cols = ['ins_dt', 'dev_id', 'coll_dt', 'coll_hour', 'is_snd', 'snd_dt', 'car_id', 'b_soc', 'b_pack_current', 'b_pack_volt', 'b_max_temp', 'b_min_temp', 'b_max_temp_modul_no', 'b_min_temp_modul_no', 'b_max_temp_pack_no', 'b_min_temp_pack_no', 'b_extern_tmep', 'b_max_cell_volt', 'b_max_cell_no', 'b_min_cell_volt', 'b_min_cell_no', 'b_assist_batt_volt', 'b_accum_charg_current_quan', 'b_accum_discharg_current_quan', 'b_accum_charg_power_quan', 'b_accum_discharg_power_quan', 'b_accum_recover_brake_quan', 'b_total_op_tm', 'b_inverter_capa_volt', 'b_moter_rpm', 'b_max_chargable_power', 'b_max_dischargable_power', 'b_cell_volt_deff', 'b_heater1_temp', 'b_soh', 'b_max_heat_cell_no', 'b_battery_remain_quan', 'b_soc_disp', 'b_coolant_inlet_temp', 'b_pra_temp', 'b_cell_balance_sts', 'b_cell_balance_cnt', 'b_slow_charg_cnt', 'b_fast_charg_cnt', 'b_accum_slow_charg_energy', 'b_accum_fast_charg_energy', 'b_battery_mod', 'b_main_relay_on_sts', 'b_main_relay_on_error_sts', 'b_batt_usable_sts', 'b_bms_alert_sts', 'b_bms_alert_error_sts', 'b_bms_fault_sts', 'b_bms_fusion_sts', 'b_bms_fusion_error_sts', 'b_opd_on_sts', 'b_opd_on_error_sts', 'b_wintermod_sts', 'b_wintermod_unInstall_sts', 'b_mcu_main_relay_off_req', 'b_mcu_control_sts', 'b_mcu_control_error_sts', 'b_fast_charg_process_sts', 'b_fast_charg_process_error_sts', 'b_charg_lamp_sts_1', 'b_charg_lamp_sts_2', 'b_charg_lamp_sts_3', 'b_fast_relay_on_sts', 'b_slow_charg_con_sts', 'b_slow_charg_con_error_sts', 'b_fast_charg_con_sts', 'm_mcu_fault_sts', 'm_mcu_torque_limit_sts', 'm_moter_controlable_sts01', 'm_mcu_engine_alert_lamp_sts', 'm_mcu_service_alert_lamp_sts', 'm_mcu_normal_sts', 'm_moter_controlable_sts02', 'm_mcu_main_relay_off_req_right', 'm_moter_pwm_sts', 'o_obc_inner_dc_volt', 'o_obc_dc_volt', 'o_chargmod_dc_volt', 'o_chargmod_dc_current', 'o_obc_fault_sts', 'o_service_lamp_req', 'o_assist_batt_soc', 'o_assist_batt_temp', 'v_car_speed', 'v_bake_a_sts', 'v_bake_b_sts', 'v_accel_pedal_depth', 'v_ev_ready_sts', 'a_inner_temp_sensor_front', 'a_extern_temp_sensor', 'a_pm_sensor', 'c_mileage', 'cc_power_send_mod_init', 'cc_power_send_mod_Charg', 'cc_power_send_mod_V2L', 'cc_power_send_mod_V2V', 'cc_evse_inout', 'cc_slow_charg_tt_cnt', 'cc_fast_charg_tt_cnt', 'cc_slow_charg_tt_hour', 'cc_fast_charg_tt_hour', 'cc_after_slow_charg_min', 'cc_after_fast_charg_min', 'ps_column_torque', 'ps_steer_angle', 'ps_column_speed', 'bc_fl_wheel_speed', 'bc_fr_wheel_speed', 'bc_rl_wheel_speed', 'bc_rr_wheel_speed', 'tp_base_front_pressure', 'tp_base_rear_pressure', 'tp_fl_pressure', 'tp_fl_temp', 'tp_fr_pressure', 'tp_fr_temp', 'tp_rl_pressure', 'tp_rl_temp', 'tp_rr_pressure', 'tp_rr_temp', 'ii_door_act_switch', 'b_cell_temp_pack_no1', 'b_cell_temp_pack_no2', 'b_cell_temp_pack_no3', 'b_cell_temp_pack_no4', 'b_cell_temp_modul_no1', 'b_cell_temp_modul_no2', 'b_cell_temp_modul_no3', 'b_cell_temp_modul_no4', 'b_modul_1_temp', 'b_modul_2_temp', 'b_modul_3_temp', 'b_modul_4_temp', 'b_modul_5_temp', 'b_modul_6_temp', 'b_modul_7_temp', 'b_modul_8_temp', 'b_modul_9_temp', 'b_modul_10_temp', 'b_modul_11_temp', 'b_modul_12_temp', 'b_modul_13_temp', 'b_modul_14_temp', 'b_modul_15_temp', 'b_modul_16_temp', 'b_modul_17_temp', 'b_modul_18_temp', 'b_modul_19_temp', 'b_modul_20_temp', 'b_modul_21_temp', 'b_modul_22_temp', 'b_modul_23_temp', 'b_modul_24_temp', 'b_cell1_volt', 'b_cell2_volt', 'b_cell3_volt', 'b_cell4_volt', 'b_cell5_volt', 'b_cell6_volt', 'b_cell7_volt', 'b_cell8_volt', 'b_cell9_volt', 'b_cell10_volt', 'b_cell11_volt', 'b_cell12_volt', 'b_cell13_volt', 'b_cell14_volt', 'b_cell15_volt', 'b_cell16_volt', 'b_cell17_volt', 'b_cell18_volt', 'b_cell19_volt', 'b_cell20_volt', 'b_cell21_volt', 'b_cell22_volt', 'b_cell23_volt', 'b_cell24_volt', 'b_cell25_volt', 'b_cell26_volt', 'b_cell27_volt', 'b_cell28_volt', 'b_cell29_volt', 'b_cell30_volt', 'b_cell31_volt', 'b_cell32_volt', 'b_cell33_volt', 'b_cell34_volt', 'b_cell35_volt', 'b_cell36_volt', 'b_cell37_volt', 'b_cell38_volt', 'b_cell39_volt', 'b_cell40_volt', 'b_cell41_volt', 'b_cell42_volt', 'b_cell43_volt', 'b_cell44_volt', 'b_cell45_volt', 'b_cell46_volt', 'b_cell47_volt', 'b_cell48_volt', 'b_cell49_volt', 'b_cell50_volt', 'b_cell51_volt', 'b_cell52_volt', 'b_cell53_volt', 'b_cell54_volt', 'b_cell55_volt', 'b_cell56_volt', 'b_cell57_volt', 'b_cell58_volt', 'b_cell59_volt', 'b_cell60_volt', 'b_cell61_volt', 'b_cell62_volt', 'b_cell63_volt', 'b_cell64_volt', 'b_cell65_volt', 'b_cell66_volt', 'b_cell67_volt', 'b_cell68_volt', 'b_cell69_volt', 'b_cell70_volt', 'b_cell71_volt', 'b_cell72_volt', 'b_cell73_volt', 'b_cell74_volt', 'b_cell75_volt', 'b_cell76_volt', 'b_cell77_volt', 'b_cell78_volt', 'b_cell79_volt', 'b_cell80_volt', 'b_cell81_volt', 'b_cell82_volt', 'b_cell83_volt', 'b_cell84_volt', 'b_cell85_volt', 'b_cell86_volt', 'b_cell87_volt', 'b_cell88_volt', 'b_cell89_volt', 'b_cell90_volt', 'b_cell91_volt', 'b_cell92_volt', 'b_cell93_volt', 'b_cell94_volt', 'b_cell95_volt', 'b_cell96_volt', 'b_cell97_volt', 'b_cell98_volt', 'b_cell99_volt', 'b_cell100_volt', 'b_cell101_volt', 'b_cell102_volt', 'b_cell103_volt', 'b_cell104_volt', 'b_cell105_volt', 'b_cell106_volt', 'b_cell107_volt', 'b_cell108_volt', 'b_cell109_volt', 'b_cell110_volt', 'b_cell111_volt', 'b_cell112_volt', 'b_cell113_volt', 'b_cell114_volt', 'b_cell115_volt', 'b_cell116_volt', 'b_cell117_volt', 'b_cell118_volt', 'b_cell119_volt', 'b_cell120_volt', 'b_cell121_volt', 'b_cell122_volt', 'b_cell123_volt', 'b_cell124_volt', 'b_cell125_volt', 'b_cell126_volt', 'b_cell127_volt', 'b_cell128_volt', 'b_cell129_volt', 'b_cell130_volt', 'b_cell131_volt', 'b_cell132_volt', 'b_cell133_volt', 'b_cell134_volt', 'b_cell135_volt', 'b_cell136_volt', 'b_cell137_volt', 'b_cell138_volt', 'b_cell139_volt', 'b_cell140_volt', 'b_cell141_volt', 'b_cell142_volt', 'b_cell143_volt', 'b_cell144_volt', 'b_cell145_volt', 'b_cell146_volt', 'b_cell147_volt', 'b_cell148_volt', 'b_cell149_volt', 'b_cell150_volt', 'b_cell151_volt', 'b_cell152_volt', 'b_cell153_volt', 'b_cell154_volt', 'b_cell155_volt', 'b_cell156_volt', 'b_cell157_volt', 'b_cell158_volt', 'b_cell159_volt', 'b_cell160_volt', 'b_cell161_volt', 'b_cell162_volt', 'b_cell163_volt', 'b_cell164_volt', 'b_cell165_volt', 'b_cell166_volt', 'b_cell167_volt', 'b_cell168_volt', 'b_cell169_volt', 'b_cell170_volt', 'b_cell171_volt', 'b_cell172_volt', 'b_cell173_volt', 'b_cell174_volt', 'b_cell175_volt', 'b_cell176_volt', 'b_cell177_volt', 'b_cell178_volt', 'b_cell179_volt', 'b_cell180_volt']
cols=['INS_DT','DEV_ID','COLL_DT','COLL_HOUR','IS_SND','SND_DT','CAR_ID','B_SOC','B_PACK_CURRENT','B_PACK_VOLT','B_MAX_TEMP','B_MIN_TEMP','B_MAX_TEMP_MODUL_NO','B_MIN_TEMP_MODUL_NO','B_MAX_TEMP_PACK_NO','B_MIN_TEMP_PACK_NO','B_EXTERN_TMEP','B_MAX_CELL_VOLT','B_MAX_CELL_NO','B_MIN_CELL_VOLT','B_MIN_CELL_NO','B_ASSIST_BATT_VOLT','B_ACCUM_CHARG_CURRENT_QUAN','B_ACCUM_DISCHARG_CURRENT_QUAN','B_ACCUM_CHARG_POWER_QUAN','B_ACCUM_DISCHARG_POWER_QUAN','B_ACCUM_RECOVER_BRAKE_QUAN','B_TOTAL_OP_TM','B_INVERTER_CAPA_VOLT','B_MOTER_RPM','B_MAX_CHARGABLE_POWER','B_MAX_DISCHARGABLE_POWER','B_CELL_VOLT_DEFF','B_HEATER1_TEMP','B_SOH','B_MAX_HEAT_CELL_NO','B_BATTERY_REMAIN_QUAN','B_SOC_DISP','B_COOLANT_INLET_TEMP','B_PRA_TEMP','B_CELL_BALANCE_STS','B_CELL_BALANCE_CNT','B_SLOW_CHARG_CNT','B_FAST_CHARG_CNT','B_ACCUM_SLOW_CHARG_ENERGY','B_ACCUM_FAST_CHARG_ENERGY','B_BATTERY_MOD','B_MAIN_RELAY_ON_STS','B_MAIN_RELAY_ON_ERROR_STS','B_BATT_USABLE_STS','B_BMS_ALERT_STS','B_BMS_ALERT_ERROR_STS','B_BMS_FAULT_STS','B_BMS_FUSION_STS','B_BMS_FUSION_ERROR_STS','B_OPD_ON_STS','B_OPD_ON_ERROR_STS','B_WINTERMOD_STS','B_WINTERMOD_UNINSTALL_STS','B_MCU_MAIN_RELAY_OFF_REQ','B_MCU_CONTROL_STS','B_MCU_CONTROL_ERROR_STS','B_FAST_CHARG_PROCESS_STS','B_FAST_CHARG_PROCESS_ERROR_STS','B_CHARG_LAMP_STS_1','B_CHARG_LAMP_STS_2','B_CHARG_LAMP_STS_3','B_FAST_RELAY_ON_STS','B_SLOW_CHARG_CON_STS','B_SLOW_CHARG_CON_ERROR_STS','B_FAST_CHARG_CON_STS','M_MCU_FAULT_STS','M_MCU_TORQUE_LIMIT_STS','M_MOTER_CONTROLABLE_STS01','M_MCU_ENGINE_ALERT_LAMP_STS','M_MCU_SERVICE_ALERT_LAMP_STS','M_MCU_NORMAL_STS','M_MOTER_CONTROLABLE_STS02','M_MCU_MAIN_RELAY_OFF_REQ_RIGHT','M_MOTER_PWM_STS','O_OBC_INNER_DC_VOLT','O_OBC_DC_VOLT','O_CHARGMOD_DC_VOLT','O_CHARGMOD_DC_CURRENT','O_OBC_FAULT_STS','O_SERVICE_LAMP_REQ','O_ASSIST_BATT_SOC','O_ASSIST_BATT_TEMP','V_CAR_SPEED','V_BAKE_A_STS','V_BAKE_B_STS','V_ACCEL_PEDAL_DEPTH','V_EV_READY_STS','A_INNER_TEMP_SENSOR_FRONT','A_EXTERN_TEMP_SENSOR','A_PM_SENSOR','C_MILEAGE','CC_POWER_SEND_MOD_INIT','CC_POWER_SEND_MOD_CHARG','CC_POWER_SEND_MOD_V2L','CC_POWER_SEND_MOD_V2V','CC_EVSE_INOUT','CC_SLOW_CHARG_TT_CNT','CC_FAST_CHARG_TT_CNT','CC_SLOW_CHARG_TT_HOUR','CC_FAST_CHARG_TT_HOUR','CC_AFTER_SLOW_CHARG_MIN','CC_AFTER_FAST_CHARG_MIN','PS_COLUMN_TORQUE','PS_STEER_ANGLE','PS_COLUMN_SPEED','BC_FL_WHEEL_SPEED','BC_FR_WHEEL_SPEED','BC_RL_WHEEL_SPEED','BC_RR_WHEEL_SPEED','TP_BASE_FRONT_PRESSURE','TP_BASE_REAR_PRESSURE','TP_FL_PRESSURE','TP_FL_TEMP','TP_FR_PRESSURE','TP_FR_TEMP','TP_RL_PRESSURE','TP_RL_TEMP','TP_RR_PRESSURE','TP_RR_TEMP','II_DOOR_ACT_SWITCH','B_CELL_TEMP_PACK_NO1','B_CELL_TEMP_PACK_NO2','B_CELL_TEMP_PACK_NO3','B_CELL_TEMP_PACK_NO4','B_CELL_TEMP_MODUL_NO1','B_CELL_TEMP_MODUL_NO2','B_CELL_TEMP_MODUL_NO3','B_CELL_TEMP_MODUL_NO4','B_MODUL_1_TEMP','B_MODUL_2_TEMP','B_MODUL_3_TEMP','B_MODUL_4_TEMP','B_MODUL_5_TEMP','B_MODUL_6_TEMP','B_MODUL_7_TEMP','B_MODUL_8_TEMP','B_MODUL_9_TEMP','B_MODUL_10_TEMP','B_MODUL_11_TEMP','B_MODUL_12_TEMP','B_MODUL_13_TEMP','B_MODUL_14_TEMP','B_MODUL_15_TEMP','B_MODUL_16_TEMP','B_MODUL_17_TEMP','B_MODUL_18_TEMP','B_MODUL_19_TEMP','B_MODUL_20_TEMP','B_MODUL_21_TEMP','B_MODUL_22_TEMP','B_MODUL_23_TEMP','B_MODUL_24_TEMP','B_CELL1_VOLT','B_CELL2_VOLT','B_CELL3_VOLT','B_CELL4_VOLT','B_CELL5_VOLT','B_CELL6_VOLT','B_CELL7_VOLT','B_CELL8_VOLT','B_CELL9_VOLT','B_CELL10_VOLT','B_CELL11_VOLT','B_CELL12_VOLT','B_CELL13_VOLT','B_CELL14_VOLT','B_CELL15_VOLT','B_CELL16_VOLT','B_CELL17_VOLT','B_CELL18_VOLT','B_CELL19_VOLT','B_CELL20_VOLT','B_CELL21_VOLT','B_CELL22_VOLT','B_CELL23_VOLT','B_CELL24_VOLT','B_CELL25_VOLT','B_CELL26_VOLT','B_CELL27_VOLT','B_CELL28_VOLT','B_CELL29_VOLT','B_CELL30_VOLT','B_CELL31_VOLT','B_CELL32_VOLT','B_CELL33_VOLT','B_CELL34_VOLT','B_CELL35_VOLT','B_CELL36_VOLT','B_CELL37_VOLT','B_CELL38_VOLT','B_CELL39_VOLT','B_CELL40_VOLT','B_CELL41_VOLT','B_CELL42_VOLT','B_CELL43_VOLT','B_CELL44_VOLT','B_CELL45_VOLT','B_CELL46_VOLT','B_CELL47_VOLT','B_CELL48_VOLT','B_CELL49_VOLT','B_CELL50_VOLT','B_CELL51_VOLT','B_CELL52_VOLT','B_CELL53_VOLT','B_CELL54_VOLT','B_CELL55_VOLT','B_CELL56_VOLT','B_CELL57_VOLT','B_CELL58_VOLT','B_CELL59_VOLT','B_CELL60_VOLT','B_CELL61_VOLT','B_CELL62_VOLT','B_CELL63_VOLT','B_CELL64_VOLT','B_CELL65_VOLT','B_CELL66_VOLT','B_CELL67_VOLT','B_CELL68_VOLT','B_CELL69_VOLT','B_CELL70_VOLT','B_CELL71_VOLT','B_CELL72_VOLT','B_CELL73_VOLT','B_CELL74_VOLT','B_CELL75_VOLT','B_CELL76_VOLT','B_CELL77_VOLT','B_CELL78_VOLT','B_CELL79_VOLT','B_CELL80_VOLT','B_CELL81_VOLT','B_CELL82_VOLT','B_CELL83_VOLT','B_CELL84_VOLT','B_CELL85_VOLT','B_CELL86_VOLT','B_CELL87_VOLT','B_CELL88_VOLT','B_CELL89_VOLT','B_CELL90_VOLT','B_CELL91_VOLT','B_CELL92_VOLT','B_CELL93_VOLT','B_CELL94_VOLT','B_CELL95_VOLT','B_CELL96_VOLT','B_CELL97_VOLT','B_CELL98_VOLT','B_CELL99_VOLT','B_CELL100_VOLT','B_CELL101_VOLT','B_CELL102_VOLT','B_CELL103_VOLT','B_CELL104_VOLT','B_CELL105_VOLT','B_CELL106_VOLT','B_CELL107_VOLT','B_CELL108_VOLT','B_CELL109_VOLT','B_CELL110_VOLT','B_CELL111_VOLT','B_CELL112_VOLT','B_CELL113_VOLT','B_CELL114_VOLT','B_CELL115_VOLT','B_CELL116_VOLT','B_CELL117_VOLT','B_CELL118_VOLT','B_CELL119_VOLT','B_CELL120_VOLT','B_CELL121_VOLT','B_CELL122_VOLT','B_CELL123_VOLT','B_CELL124_VOLT','B_CELL125_VOLT','B_CELL126_VOLT','B_CELL127_VOLT','B_CELL128_VOLT','B_CELL129_VOLT','B_CELL130_VOLT','B_CELL131_VOLT','B_CELL132_VOLT','B_CELL133_VOLT','B_CELL134_VOLT','B_CELL135_VOLT','B_CELL136_VOLT','B_CELL137_VOLT','B_CELL138_VOLT','B_CELL139_VOLT','B_CELL140_VOLT','B_CELL141_VOLT','B_CELL142_VOLT','B_CELL143_VOLT','B_CELL144_VOLT','B_CELL145_VOLT','B_CELL146_VOLT','B_CELL147_VOLT','B_CELL148_VOLT','B_CELL149_VOLT','B_CELL150_VOLT','B_CELL151_VOLT','B_CELL152_VOLT','B_CELL153_VOLT','B_CELL154_VOLT','B_CELL155_VOLT','B_CELL156_VOLT','B_CELL157_VOLT','B_CELL158_VOLT','B_CELL159_VOLT','B_CELL160_VOLT','B_CELL161_VOLT','B_CELL162_VOLT','B_CELL163_VOLT','B_CELL164_VOLT','B_CELL165_VOLT','B_CELL166_VOLT','B_CELL167_VOLT','B_CELL168_VOLT','B_CELL169_VOLT','B_CELL170_VOLT','B_CELL171_VOLT','B_CELL172_VOLT','B_CELL173_VOLT','B_CELL174_VOLT','B_CELL175_VOLT','B_CELL176_VOLT','B_CELL177_VOLT','B_CELL178_VOLT','B_CELL179_VOLT','B_CELL180_VOLT','B_CELL181_VOLT','B_CELL182_VOLT','B_CELL183_VOLT','B_CELL184_VOLT','B_CELL185_VOLT','B_CELL186_VOLT','B_CELL187_VOLT','B_CELL188_VOLT','B_CELL189_VOLT','B_CELL190_VOLT','B_CELL191_VOLT','B_CELL192_VOLT','RSV_INT_1','RSV_INT_2','RSV_INT_3','RSV_INT_4','RSV_INT_5','RSV_FLOAT_1','RSV_FLOAT_2','RSV_FLOAT_3','RSV_FLOAT_4','RSV_FLOAT_5','VIN']

def printProgressBar(iteration, total, prefix = 'Progress', suffix = 'Complete',\
                      decimals = 1, length = 50, fill = '█'): 
    # 작업의 진행상황을 표시
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' %(prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()
    if iteration == total:
        print()


def recursive_search_dir(_nowDir, _filelist):
    dir_list = []  # 현재 디렉토리의 서브디렉토리가 담길 list
    try:
        f_list = os.listdir(_nowDir)
    except FileNotFoundError:
        print("\n"+_nowDir)
        print("\n(CSV ID별 분류 대상 파일이 존재하지 않습니다.)")
        sys.exit(1)

    for fname in f_list:
        if os.path.isdir(_nowDir + "/" + fname):
            dir_list.append(_nowDir + "/" + fname)
        elif os.path.isfile(_nowDir + "/" + fname):
            file_extension = os.path.splitext(fname)[1]
            if file_extension == ".csv" or file_extension == ".CSV":  # csv
                _filelist.append(_nowDir + "/" + fname)

    for toDir in dir_list:
        recursive_search_dir(toDir, _filelist)
        

@ray.remote
def split_by_id(csv_file, _id, out_path, _existing_carid):
    skip_rows = 0
    batch_size = 500000
    total_lines = 0
    while True:
        try:
            df = pd.read_csv(csv_file, low_memory=False, names=cols, usecols=range(338), header=None, skiprows=skip_rows, nrows=batch_size)
            carid_df = pd.read_csv(_existing_carid)
            if len(df) == 0:
                break
            total_lines += len(df)
            skip_rows += batch_size
            for carid in list(df[_id].unique()):
                new_df = df[df[_id] == carid].copy()
                car_type = csv_file.split('/')[-2]
                if car_type != 'EV6' and car_type != 'IONIQ5' and car_type != 'KONA' and car_type != 'G80' and car_type != 'NIRO':
                    for i in range(len(carid_df)):
                        if carid == carid_df.iloc[i]['car_id']:
                            car_type = carid_df.iloc[i]['car_type']
                        else: i += 1

                new_df['car_type'] = car_type
                tout_path = out_path + '/' + car_type + '/'
                if not os.path.isdir(tout_path):
                    try:
                        os.makedirs(tout_path)
                    except:
                        pass
                if os.path.isfile(tout_path + carid + '.csv'):
                    new_df.to_csv(tout_path + carid + '.csv', mode='a', header=False, index=False)
                else:
                    new_df.to_csv(tout_path + carid + '.csv', mode='w', header=True, index=False)
        except Exception as e:
            print('\n==========================================================================')
            print('%s' % csv_file)
            print(e)
            print(traceback.format_exc())
            print('==========================================================================\n')
            break
    return total_lines

if __name__ == '__main__':

    pn = int(sys.argv[1].replace('\r', ''))
    csv_path = sys.argv[2].replace('\r', '')
    out_path = sys.argv[3].replace('\r', '')
    _id = sys.argv[4].replace('\r', '')
    _existing_carid = './existing_carid_list.csv'

    if pn <= 0:
        ray.init()
    else:    
        ray.init(num_cpus=pn)
    
    if csv_path[-1] == '/':
        csv_path = csv_path[:-1]
    if out_path[-1] == '/':
        out_path = out_path[:-1]

    print (" CSV Dir location = ", csv_path)
    csv_list = []

    proc_start_time = time.time()

    print('\n======================================================')
    print('CSV ID별 분류 시작')
    print('======================================================')
    print('CSV 파일 목록 불러오는 중..')
    recursive_search_dir(csv_path, csv_list)
    csv_len = len(csv_list)
    print('총 CSV파일 수 : {}'.format(csv_len))

    obj_id_list = []
    print('\nCSV ID별 분류 중..')
    for csv_file in csv_list:
        obj_id_list.append(split_by_id.remote(csv_file, _id, out_path, _existing_carid))

    cnt=0
    total_rows = 0
    while len(obj_id_list):
        printProgressBar(cnt, csv_len)
        done, obj_id_list = ray.wait(obj_id_list)
        total_rows += ray.get(done[0])
        cnt+=1
    printProgressBar(cnt, csv_len)

    print("\n총 처리한 CSV row 수 : {}".format(total_rows))
    print("출력 파일 경로 : {}".format(out_path))
    print('total running time : {:.2f} sec'.format(time.time()-proc_start_time))
