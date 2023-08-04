import numpy as np

from cam_config import *
def norm_data_IMU(acc_data,gyro_data):
    acc_data=acc_data[acc_data.shape[0]-gyro_data.shape[0]:]
    return acc_data,gyro_data
def get_spilt_data():
    base_path=r"D:\all_code\cam_code\data\original\test1"
    X_all=np.zeros((0,6,spilt_num))
    y_all = np.zeros((0,1))
    for i in range(3):
        for j in range(3):
            filepath = os.path.join(base_path, str(i),str(j)+".xls")
            acc_data = (pd.read_excel(filepath, sheet_name="Accelerometer")).values[:, 1:4]  # *************************
            gyro_data = (pd.read_excel(filepath, sheet_name="Gyroscope")).values[:, 1:4]  # *************************
            acc_data, gyro_data = norm_data_IMU(acc_data, gyro_data)
            IMU_data = np.hstack((acc_data, gyro_data))
            length_temp=IMU_data.shape[0]
            index=0
            while(index+spilt_num<length_temp):
                new_temp=np.expand_dims(IMU_data[index:index+spilt_num].T, axis=0)
                X_all = np.concatenate((X_all, new_temp), axis=0)
                y_all = np.vstack((y_all, i))
                index+=random.randint(0, step_num)
    save_path = r"D:\all_code\cam_code\data\train\test1"
    save_path_X = os.path.join(save_path, "X_all.npy")
    save_path_y = os.path.join(save_path, "y_all.npy")

    X_all=X_all.astype(np.float32)
    y_all=(y_all.squeeze()).astype(np.longlong)
    # 存数据
    np.save(save_path_X, X_all)
    np.save(save_path_y, y_all)
    # 读数据
    X_all = np.load(save_path_X)
    y_all = np.load(save_path_y)
    return X_all,y_all
def get_test_data(save_path,index):
    filepath = os.path.join(save_path, index+ ".xls")
    print(filepath)
    acc_data = (pd.read_excel(filepath, sheet_name="Accelerometer")).values[:, 1:4]  # *************************
    gyro_data = (pd.read_excel(filepath, sheet_name="Gyroscope")).values[:, 1:4]  # *************************
    acc_data, gyro_data = norm_data_IMU(acc_data, gyro_data)
    IMU_data = np.hstack((acc_data, gyro_data))
    IMU_data=IMU_data.T
    return IMU_data
def get_cam_2D(weight,last_conv_value,out):

    weight=weight.detach().numpy()
    last_conv_value=last_conv_value.detach().numpy()
    out=out.detach().numpy()
    max_index=out.squeeze(0).argmax()
    weight=weight[max_index]
    result=np.zeros((last_conv_value.shape[-2],last_conv_value.shape[-1]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i,j]= np.dot(last_conv_value[0,:,i,j],weight )
    return result


def get_cam_1D(weight, last_conv_value, out):
    weight = weight.detach().numpy()
    last_conv_value = last_conv_value.detach().numpy()
    if last_conv_value.shape[0] == 1:
        last_conv_value = np.squeeze(last_conv_value)
    out = out.detach().numpy()


    result = np.zeros((out.shape[-1],last_conv_value.shape[-1]))

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            weight_temp=weight[i]
            result[i,j] = np.dot(last_conv_value[:, j], weight_temp)
    return result
if __name__ == '__main__':

    X_all,y_all=get_spilt_data()
    print(X_all.shape,y_all.shape)
    print(y_all)