import numpy as np
import h5py
from scipy import signal

from correlation_segmentation import Correlation_Segmentation

full_tasks = ['fetch_recorded_demo_1730997119.h5', 'fetch_recorded_demo_1730997530.h5', 'fetch_recorded_demo_1730997735.h5', 'fetch_recorded_demo_1730997956.h5']
ground_truths = [[0, 1125, 2591, 3986, 5666], [0, 1812, 3844, 5732, 7090], [0, 1965, 4178, 6427, 7904], [0, 1898, 4081, 5442, 6829]]
sub_tasks = ['fetch_recorded_demo_1730996323.h5', 'fetch_recorded_demo_1730996415.h5', 'fetch_recorded_demo_1730996653.h5', 'fetch_recorded_demo_1730996760.h5', 'fetch_recorded_demo_1730996917.h5']

def read_data(fname):
    hf = h5py.File('../h5_files/' + fname, 'r')
    #print(list(hf.keys()))
    js = hf.get('joint_state_info')
    joint_time = np.array(js.get('joint_time'))
    joint_pos = np.array(js.get('joint_positions'))
    joint_vel = np.array(js.get('joint_velocities'))
    joint_eff = np.array(js.get('joint_effort'))
    joint_data = [joint_time, joint_pos, joint_vel, joint_eff]

    tf = hf.get('transform_info')
    tf_time = np.array(tf.get('transform_time'))
    tf_pos = np.array(tf.get('transform_positions'))
    tf_rot = np.array(tf.get('transform_orientations'))
    tf_data = [tf_time, tf_pos, tf_rot]
    #print(tf_pos)

    # wr = hf.get('wrench_info')
    # wrench_time = np.array(wr.get('wrench_time'))
    # wrench_frc = np.array(wr.get('wrench_force'))
    # wrench_trq = np.array(wr.get('wrench_torque'))
    # wrench_data = [wrench_time, wrench_frc, wrench_trq]

    gp = hf.get('gripper_info')
    gripper_time = np.array(gp.get('gripper_time'))
    gripper_pos = np.array(gp.get('gripper_position'))
    gripper_data = [gripper_time, gripper_pos]

    hf.close()

    # return joint_data, tf_data, wrench_data, gripper_data
    return joint_data, tf_data, gripper_data
    
def smooth(data):
    data_smooth = []
    _, n_dims = np.shape(data)
    for i in range(n_dims):
        data_smooth.append(signal.savgol_filter(data[:, i], 301, 2))
    return np.transpose(np.vstack((data_smooth)))
    
def minmax_norm(data):
    dmin = np.min(data)
    dmax = np.max(data)
    if dmax - dmin == 0:
        return (data - dmin)
    return (data - dmin) / (dmax - dmin)
    
def zscore_norm(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return (data - mean)
    return (data - mean) / std
    
def feature_norm(data):
    data_norm = []
    _, n_dims = np.shape(data)
    for i in range(n_dims):
        data_norm.append(zscore_norm(data[:, i]))
    return np.transpose(np.vstack((data_norm)))
    
def robust_norm(data):
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    if iqr == 0:
        return (data - median)
    return (data - median) / iqr
    
def nothing(data):
    return data
    
if __name__ == '__main__':

    # load data
    full_task_data = []
    for i in range(len(full_tasks)):
        joint_data, tf_data, gripper_data = read_data(full_tasks[i])
        full_task_data.append(tf_data[1])
        
    subtask_data = []
    for i in range(len(sub_tasks)):
        joint_data, tf_data, gripper_data = read_data(sub_tasks[i])
        subtask_data.append(tf_data[1])
    
    # segment for smoothing, normalization, sim_metric
    smoothing = [smooth]
    norms = [nothing, minmax_norm, feature_norm, zscore_norm, robust_norm]
    sim_metrics = ['CCS', 'SSE', 'COS']
    
    accs = np.zeros((4*3*5*2, 6))
    inds = np.zeros((4*3*5*2, 5))
    idx = 0
    
    for i in range(len(smoothing)):
        for j in range(len(norms)):
            for k in range(len(sim_metrics)):
                for l in range(len(full_task_data)):
                    f_data = norms[j](smoothing[i](full_task_data[l]))
                    print(np.shape(f_data))
                    s_data = [norms[j](smoothing[i](subtask_data[m])) for m in range(len(subtask_data))]
                    print(np.shape(s_data[0]))
                    cs = Correlation_Segmentation(f_data, s_data, sim_metrics[k])
                    classes = cs.segment3()
                    print(classes)
                    gt = ground_truths[l]
                    st1_tru = sum(classes[gt[0]:gt[1]] == 0)
                    st2_tru = sum(classes[gt[1]:gt[2]] == 1)
                    st3_tru = sum(classes[gt[2]:gt[3]] == 2)
                    st4_tru = sum(classes[gt[3]:gt[4]] == 3)
                    st5_tru = sum(classes[gt[4]:] == 4)
                    
                    total_true = st1_tru + st2_tru + st3_tru + st4_tru + st5_tru
                    
                    st1_acc = st1_tru / (gt[1] - gt[0])
                    st2_acc = st2_tru / (gt[2] - gt[1])
                    st3_acc = st3_tru / (gt[3] - gt[2])
                    st4_acc = st4_tru / (gt[4] - gt[3])
                    st5_acc = st5_tru / (len(full_task_data[l]) - gt[4])
                    
                    total_acc = total_true / len(full_task_data[l])
                    
                    print(i, j, k, l, st1_acc, st2_acc, st3_acc, st4_acc, st5_acc, total_acc)
                    accs[idx, :] = np.array([st1_acc, st2_acc, st3_acc, st4_acc, st5_acc, total_acc])
                    inds[idx, :] = cs.inds
                    
                    idx = idx + 1
                    np.savetxt('smooth_seg_acc4.txt', accs)
                    np.savetxt('smooth_seg_inds4.txt', inds)
    print(accs)
    print(inds)