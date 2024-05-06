import torch
import pytorch3d.transforms as transform
import numpy as np 
class HandPoseSuggest():
    def __init__(self) -> None:
        self.suggest
        self.handpose
        pass
class ShadowHandGrounding():
    def __init__(self) -> None:
        self.output_sequence=['WRJ2', 'WRJ1', 'FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 'LFJ1', 'THJ5', 'THJ4', 'THJ3', 'THJ2', 'THJ1']
        pass
    def mapping(self,suggest_joints):
        # joints: hand orientation euler XYZ [3]
        #         each finger from thumb to pinky finger *4 from root to fingertip 绕轴转动+三个内侧反转 [20]
        print('suggest_joints',suggest_joints)
        joints=(np.array(suggest_joints[3:]).reshape(-1))*np.pi/180
        print(joints)
        joint_mapping={}
        joint_mapping['WRJ2']=0 
        # 手腕左右运动
        joint_mapping['WRJ1']=-0.5
        # 最大值手腕内翻，最小值手腕外翻
        
        # thumb finger 
        joint_mapping['THJ5']=joints[1]
        joint_mapping['THJ4']=joints[0]
        joint_mapping['THJ3']=joints[0]*0.3
        joint_mapping['THJ2']=joints[2]
        joint_mapping['THJ1']=joints[3]
        
        # index finger
        joint_mapping['FFJ4']=-joints[4]
        joint_mapping['FFJ3']=joints[5]
        joint_mapping['FFJ2']=joints[6]
        joint_mapping['FFJ1']=joints[7]
        
        # middle finger 
        joint_mapping['MFJ4']=-joints[8]
        joint_mapping['MFJ3']=joints[9]
        joint_mapping['MFJ2']=joints[10]
        joint_mapping['MFJ1']=joints[11]
        
        # ring finger 
        joint_mapping['RFJ4']=-joints[12]
        joint_mapping['RFJ3']=joints[14]
        joint_mapping['RFJ2']=joints[14]
        joint_mapping['RFJ1']=joints[15]
        
        # pinky finger 
        joint_mapping['LFJ5']=-joints[16]*0.3
        joint_mapping['LFJ4']=-joints[16]
        joint_mapping['LFJ3']=joints[17]
        joint_mapping['LFJ2']=joints[18]
        joint_mapping['LFJ1']=joints[19]
        
        joints_in_sequence=np.array([joint_mapping[joint_name] for joint_name in self.output_sequence]).reshape((1,-1))
        for idx in range(3):
            if suggest_joints[idx]==180.:
                suggest_joints[idx]=179.
            elif suggest_joints[idx]==-180.:
                suggest_joints[idx]=-179.
        palm_ori=transform.euler_angles_to_matrix(suggest_joints[:3]*torch.pi/180,convention='XYZ')
        return palm_ori,joints_in_sequence

