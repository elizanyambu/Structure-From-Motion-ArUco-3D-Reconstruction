# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:44:01 2021

@author: tkw
"""
import glob
import os
import pathlib
import cv2
import cv2.aruco as aruco
import imutils
import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric


class twoCameraIntelligence:
    def __init__(self, camera1,camera2,path1,path2, arucomarker1=24,arucomarker2=70):
        
        self.arucomarker1=arucomarker1
        self.arucomarker2=arucomarker2
        self.path1=path1
        self.path2=path2

        self.image1=arucomarker2
        self.image2=arucomarker2

        self.arucomarker1Cam1=np.array([[0,0]])
        self.arucomarker2Cam1=np.array([[0,0]])
        self.arucomarker1Cam2=np.array([[0,0]])
        self.arucomarker2Cam2=np.array([[0,0]])

        self.camera1 = camera1 
        self.camera2 = camera2 
        self.realdistance=self.getRealDistanceAruco()
        path='./output/CalibrationMatrices.yml'
        if os.path.isfile(path):
            cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
            self.ProjectionMatrix1  = cv_file.getNode('ProjectionMatrix1').mat()
            self.ProjectionMatrix2 = cv_file.getNode('ProjectionMatrix2').mat()
            self.uptoscalevalue = cv_file.getNode('uptoscalevalue').real()
            self.Trans= cv_file.getNode('TranslationCameras').mat()
            self.Rot= cv_file.getNode('RotationCameras').mat()
            self.E= cv_file.getNode('EssentialMatrix').mat()

        else:
            self.calibrateCameras()

        
    def getRealDistanceAruco(self):
        if os.path.isfile('./output/ArucoDistance.yml'):
            cv_file = cv2.FileStorage("./output/ArucoDistance.yml", cv2.FILE_STORAGE_READ)
            distance  = cv_file.getNode('distance').real()
            self.arucomarker1Cam1=cv_file.getNode('arucocenter1Cam1').mat()
            self.arucomarker2Cam1=cv_file.getNode('arucocenter2Cam1').mat()
            self.arucomarker1Cam2=cv_file.getNode('arucocenter1Cam2').mat()
            self.arucomarker2Cam2=cv_file.getNode('arucocenter2Cam2').mat()
            
        else:
            eucldistmatrix1,arucodf_1=self.camera1.extract_arucomarkers(self.path1)
            eucldistmatrix2,arucodf_2=self.camera2.extract_arucomarkers(self.path2)
            c11=arucodf_1[arucodf_1.id==self.arucomarker1]
            c12=arucodf_1[arucodf_1.id==self.arucomarker2]
            c21=arucodf_2[arucodf_2.id==self.arucomarker1]
            c22=arucodf_2[arucodf_2.id==self.arucomarker2]
    
            self.arucomarker1Cam1=np.array([[c11.iloc[0].cx,c11.iloc[0].cy]])
            self.arucomarker2Cam1=np.array([[c12.iloc[0].cx,c12.iloc[0].cy]])
            self.arucomarker1Cam2=np.array([[c21.iloc[0].cx,c21.iloc[0].cy]])
            self.arucomarker2Cam2=np.array([[c22.iloc[0].cx,c22.iloc[0].cy]])
            
            distance1=eucldistmatrix1[self.arucomarker1][self.arucomarker2]
            distance2=eucldistmatrix2[self.arucomarker1][self.arucomarker2]
            distance= (distance1+distance2)/2
            cv_file = cv2.FileStorage("./output/ArucoDistance.yml", cv2.FILE_STORAGE_WRITE)
            cv_file.write('distance',distance)
            cv_file.write('arucocenter1Cam1',self.arucomarker1Cam1)
            cv_file.write('arucocenter2Cam1',self.arucomarker2Cam1)
            cv_file.write('arucocenter1Cam2',self.arucomarker1Cam2)
            cv_file.write('arucocenter2Cam2',self.arucomarker2Cam2)
            cv_file.release()
        return distance


    def calibrateCameras(self):

        img1=cv2.imread(self.path1)
        img2=cv2.imread(self.path2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        (kps1, des1) = sift.detectAndCompute(gray1, None)
        print("# kps: {}, descriptors: {}".format(len(kps1), des1.shape))
        sift = cv2.xfeatures2d.SIFT_create()
        (kps2, des2) = sift.detectAndCompute(gray2, None)
        print("# kps: {}, descriptors: {}".format(len(kps2), des2.shape))
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        features1 = []
        features2 = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                features1.append(kps1[m.queryIdx].pt)
                features2.append(kps2[m.trainIdx].pt)
        print(len(features1))
        print(len(features2))
        pts1 = np.int32(features1)
        pts2 = np.int32(features2)     
        pts1=np.append(pts1, self.arucomarker1Cam1, axis=0)
        pts1=np.append(pts1,  self.arucomarker2Cam1, axis=0)
        pts2=np.append(pts2,  self.arucomarker1Cam2, axis=0)
        pts2=np.append(pts2,  self.arucomarker2Cam2, axis=0)
        
        f_avg = (self.camera1.IntrinsicMatrix[0, 0] + self.camera2.IntrinsicMatrix[0, 0]) / 2
        # Normalize for Esential Matrix calaculation
        pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=self.camera1.IntrinsicMatrix, distCoeffs=None)#self.camera1.DistortionVector)
        pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=self.camera2.IntrinsicMatrix, distCoeffs=None)#self.camera2.DistortionVector)
        E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3/f_avg)
        points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
        M_r = np.hstack((R, t))
        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        P1 = np.dot(self.camera1.IntrinsicMatrix,  M_l)
        P2 = np.dot(self.camera2.IntrinsicMatrix,  M_r)
        point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(pts1, axis=1), np.expand_dims(pts2, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        self.ProjectionMatrix1=P1
        self.ProjectionMatrix2=P2 
        self.Rot=R
        self.Trans=t
        self.E=E
        print(point_4d)
        rep_error = []
        print(point_4d)
        points_3d=np.transpose(point_4d_hom)
        for idx, pt_3d in enumerate(points_3d):
            pt_2d = np.array([pts1[idx][0], pts1[idx][1]])
            reprojected_pt = np.dot(self.ProjectionMatrix1, pt_3d)
            reprojected_pt /= reprojected_pt[2]
            print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
            rep_error.append(pt_2d - reprojected_pt[0:2])
        #print(rep_error)
        points_3d = point_4d[:3, :].T
        path="./output/CalibrationMatrices.yml"
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('ProjectionMatrix1',self.ProjectionMatrix1)
        cv_file.write('ProjectionMatrix2',self.ProjectionMatrix2) 
        cv_file.write('EssentialMatrix',self.E)     
        cv_file.write('RotationCameras',self.Rot)
        cv_file.write('TranslationCameras',self.Trans)     
        #points_3d=np.transpose(points_3d)
        x=points_3d[-1][0]-points_3d[-2][0]
        y=points_3d[-1][1]-points_3d[-2][1]
        z=points_3d[-1][2]-points_3d[-2][2]
        eucl=np.sqrt((x*x)+(y*y)+(z*z))     
        self.uptoscalevalue=eucl
        cv_file.write('uptoscalevalue',self.uptoscalevalue)
        cv_file.release()

    def projectpoint(self,df1,df2):
        # before go into this function one should check if the feature exits in the two pictures.
        left=df1.to_numpy()
        points_3d = cv2.triangulatePoints(self.ProjectionMatrix1, self.ProjectionMatrix2, np.transpose(df1.to_numpy()), np.transpose(df2.to_numpy()))
        points_3d=np.transpose(points_3d)
        img2 = cv2.imread(self.path1)
        rep_error = []
        for idx, pt_3d in enumerate(points_3d):
            pt_2d = np.array([left[idx][0], left[idx][1]])
            reprojected_pt = np.dot(self.ProjectionMatrix1, pt_3d)
            reprojected_pt /= reprojected_pt[2]
        
            if len(img2)>1000:
                scalecircleplot=10
            elif len(img2)>800 & len(img2)<1500:
                scalecircleplot=6
            else:
                scalecircleplot=4
                
            cv2.circle(img2,(int(reprojected_pt[0]),int(reprojected_pt[1])), scalecircleplot, (255,0,0), -1) 
            cv2.circle(img2,(int(pt_2d[0]),int(pt_2d[1])), scalecircleplot, (0,0,255), -1)
            rep_error.append(pt_2d - reprojected_pt[0:2])        
        imS = cv2.resize(img2, (800, 600)) # Resize imagewe
        cv2.imshow('Projection',imS)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def evaluate(self,df1,df2):
        points_3d = cv2.triangulatePoints(self.ProjectionMatrix1, self.ProjectionMatrix2, np.transpose(df1.to_numpy()), np.transpose(df2.to_numpy()))        
        points_3d /= points_3d[3]
        points_3d=np.transpose(points_3d)
        points_3d=points_3d*(self.realdistance/self.uptoscalevalue)
        x=points_3d[0][0]-points_3d[1][0]
        y=points_3d[0][1]-points_3d[1][1]
        z=points_3d[0][2]-points_3d[1][2]
        eucl=np.sqrt((x*x)+(y*y)+(z*z))
        print(eucl)

class IndividualCamera():
    def __init__(self, id, resolution):
        self.id=id# CAM1-CAM2-CAM3
        self.IntrinsicMatrix = 0
        self.DistortionVector = 0
        self.resolution = resolution

        #checkerboard calibration parameters
        self.dir_path='./input/'+self.id+'_Checkerboardpics/'+self.resolution
        self.square_size= 2.4   
        self.width=9 
        self.height=6
        path='./output/'+self.id+".yml"
        if os.path.isfile(path):
            self.load_coefficients()
        else:
            self.calibrate_chessboard()

            
    def calibrate_chessboard(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.height * self.width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        objp = objp * self.square_size
        objpoints = []  
        imgpoints = []  
        for fname in glob.glob(self.dir_path+'/*.jpg'): 
            print(fname)    
            gray =cv2.imread(str(fname),0)
            ret, corners = cv2.findChessboardCorners(gray, (self.width, self.height), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.IntrinsicMatrix = mtx
        self.DistortionVector = dist
        self.save_coefficients()
    
    def save_coefficients(self):
        path='./output/'+self.id+".yml"
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('IntrinsicMat',self.IntrinsicMatrix)
        cv_file.write('DistortionVec',self.DistortionVector)
        cv_file.release()
        
    def load_coefficients(self):
        path='./output/'+self.id+".yml"
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        self.IntrinsicMatrix  = cv_file.getNode('IntrinsicMat').mat()
        self.DistortionVector = cv_file.getNode('DistortionVec').mat()
          
    def extract_arucomarkers(self,imagelocation,markerSize = 5, totalMarkers=250,draw=True):        
        img=cv2.imread(imagelocation)
        key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(key)
        arucoParam = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParam,cameraMatrix=self.IntrinsicMatrix,distCoeff=self.DistortionVector)
        i=0
        df_centers = pd.DataFrame()
        for corner in corners:
                centerX = (corner[0][0][0] + corner[0][1][0] + corner[0][2][0] + corner[0][3][0]) / 4
                centerY = (corner[0][0][1] + corner[0][1][1] + corner[0][2][1] + corner[0][3][1]) / 4
                center = (int(centerX), int(centerY))

                cv2.circle(img,center, int(len(img[0])/50), (255,0,0), -1)
                cv2.circle(img,center, int(len(img[0])/50), (255,0,0), -1)
                print("id: {0}, x:{1}, y:{2}".format(ids[i],centerX,centerY))
                df_centers=df_centers.append({'id':ids[i][0],'cx':centerX,'cy':centerY}, ignore_index=True)
                print("id: {0}, x:{1}, y:{2}".format(ids[i],centerX,centerY))
                fontScale = 3
                color = (120, 255, 255)
                thickness = 8
                cv2.putText(img,"id={0}, X={1},Y={2}".format(ids[i][0], centerX, centerY), (int(centerX), int(centerY)), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA, False)
                print(ids[i][0])
                i=i+1
        df = pd.DataFrame()
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.2, self.IntrinsicMatrix,self.DistortionVector)
            print(tvec)
            df=df.append({'id':ids[i][0],'x':100*tvec[0][0][0],'y':100*tvec[0][0][1],'z':100*tvec[0][0][2]}, ignore_index=True)
            aruco.drawDetectedMarkers(img, corners)
            aruco.drawAxis(img, self.IntrinsicMatrix, self.DistortionVector, rvec, tvec, 0.1)  # Draw Axis
    
        dist = DistanceMetric.get_metric('euclidean')
        eucldistmatrix=pd.DataFrame(dist.pairwise(df[['x','y','z']].to_numpy()),  columns=df.id.unique(), index=df.id.unique())
        print(eucldistmatrix)
        if draw:
            while True:
                imS = cv2.resize(img, (800, 600)) # Resize imagewe
                cv2.imshow("output", imS)   
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        cv2.destroyAllWindows()
        return [eucldistmatrix,df_centers]

def findtennisballs(path):
    img = cv2.imread(path)
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts  = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts  = imutils.grab_contours(cnts)
    center = None	# only proceed if at least one contour was found
    arrayBallcenters=pd.DataFrame()
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        if radius>int(len(img[0])/70):
                arrayBallcenters=arrayBallcenters.append({'x':int(cX),'y':int(cY)} , ignore_index=True)
                cv2.circle(img, (int(cX), int(cY)), int(radius),(0, 0, 255), 5)
                cv2.putText(img, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)

    arrayBallcenters=arrayBallcenters.sort_values(by=['x'])
    arrayBallcenters = arrayBallcenters.reset_index(drop=True)    
    print(arrayBallcenters)
    imS = cv2.resize(img, (800, 600)) # Resize imagewe
    cv2.imshow('image',imS)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return arrayBallcenters


#############################################################################

def deletefiles():
    
    dir_name = "./output/"
    test = os.listdir(dir_name)
    
    for item in test:
        if item.endswith(".yml"):
            os.remove(os.path.join(dir_name, item))

def main():
    resolution={
        1:'640_480',
        2:'2016_1512',
        3:'4032_3024'}

    choiceCam1=1
    choiceCam2=2
    deletefiles()
    
    Cam1=IndividualCamera('CAM1',resolution[choiceCam1] )
    Cam2=IndividualCamera('CAM2',resolution[choiceCam2] )
    
    path1="./input/CalibrationImages/"+resolution[choiceCam1]+"/CAM1_Snapshot.JPG"
    path2="./input/CalibrationImages/"+resolution[choiceCam2]+"/CAM2_Snapshot.JPG"
    
    twocameras= twoCameraIntelligence(Cam1,Cam2,path1,path2, arucomarker1=70,arucomarker2=24)
    ##############################################################################

    outcam1=findtennisballs(path1)
    outcam2=findtennisballs(path2)
    ##############################################################################
    if (len(outcam1)==len(outcam2) & len(outcam1)> 0 ):
        print('ok')
        twocameras.evaluate(outcam1,outcam2)
        twocameras.projectpoint(outcam1,outcam2)
    ##############################################################################


if __name__ == "__main__":
    main()


