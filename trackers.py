import os
import cv2
import json
import tools
import numpy as np
import settings
from affine_estimator import get_affine


class ObjectTracker():
    """docstring for ObjectTracker"""
    def __init__(self, name, method, roi_size, roi_init_mode):
        self.name = name
        self.method = method
        self.roi_size = roi_size
        self.prev_cXY = None, None
        self.cXY = None, None
        self.history = []
        self.roi_init_mode = roi_init_mode

    def __init_roi__(self, prev_frame):
        # Initialize ROI coordinates manually by user input
        if self.roi_init_mode == 'manual':
            self.cXY = tools.cXY_from_click(prev_frame)
            self.prev_cXY = self.cXY
            if not self.prev_cXY[0]:
                return False, '[ERROR] ROI was not Initialized (in {})'.format(self.name)
            else:
                cv2.destroyAllWindows()
                return True, '[INFO] ROI was Initialized (in {})'.format(self.name)

        # init ROI center automatically by frame differencing
        elif self.roi_init_mode == 'auto':
            self.cXY = tools.frame_diff_detector(prev_frame, frame)
            self.prev_cXY = self.cXY

        else:
            return False, '[ERROR] ROI initialization mode unknown (in {})'.format(self.name)

    def track(self, frame):
        # get only the ROI from the current frame
        window = tools.get_roi(frame, self.prev_cXY)
        
        # segmentate the ant inside the ROI
        ant_mask = tools.get_ant_mask(window)

        # alter the blue channel in ant-related pixels
        window[:,:,0] = ant_mask

        # update the roi center using current ant coordinates
        self.cXY = tools.update_roi_center(ant_mask, self.prev_cXY)

        # update data
        self.history.append(self.cXY)
        self.prev_cXY = self.cXY
  


class CameraTracker():
    """docstring for CameraTracker"""
    def __init__(self, scale=1):
        self.scale = scale
        self.history = []
        self.mse = []

    def compute_roi(self, dim):
        self.dim = dim
        self.roi = tools.get_main_region(dim) # TODO: Update this to avoid relying on tools

    def track(self, prev_frame, frame, roi_array):
        # track the floor
        mask = tools.mask2track(self.dim, roi_array)
        p_good, aff_params, err = get_affine(prev_frame, frame, self.roi, mask)
        self.features = p_good[1:]

        if err is None:
            return False, 'No matrix was estimated'

        self.history.append(aff_params)
        self.mse.append(err)

        return True, 'Camera Tracked'
        

class TrackingScenario():
    """docstring for TrackingScenario"""
    def __init__(self, object_trackers, camera_tracker=None, undistorter=None):
        self.object_trackers = object_trackers   
        self.camera_tracker = camera_tracker 
        self.iteration_counter = 0
        self.auto_mode = True
        self.undistorter = undistorter
        self.enabled = True

    def __digest_video_path(self, video_path):
        # TODO: Validate the path
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)                        # create capture object
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total number of frames in the video file
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)                      # frames per seconds
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))           # frame width
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))          # frame height
        self.dim = (self.w, self.h)

        self.first_frame = 0

    def __undistort__(self, frame):
        if self.undistorter:
            frame = self.undistorter.fix(frame)
        return frame

    def __first_iteration__(self, start_in_frame):
        # Start processing frams at the given index
        if start_in_frame:
            self.first_frame = start_in_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_in_frame)

        # Capture the first frame to process
        ret, prev_frame = self.cap.read()
        
        # correct spherical distortion
        self.prev_frame = self.__undistort__(prev_frame)

        # Initialize the roi of all the trackers
        for otrack in self.object_trackers:
            retval, message = otrack.__init_roi__(self.prev_frame)
            if not retval:
                return retval, message

        # Initialize the region of the camera tracker
        if self.camera_tracker:
            self.camera_tracker.compute_roi(self.dim)

        # Increase the iteration counter
        self.iteration_counter += 1

        return True, '[INFO] All trackers were initialized'

    def keyboard_controller(self):
        # keyboard events
        wait_key = 0 if not self.auto_mode else 10

        k = cv2.waitKey(wait_key) & 0xff
        if k == ord('m'):
            self.auto_mode = not self.auto_mode

        if k == ord('s'):
            self.__export_data__()

        elif k == ord('q'):
            self.enabled = False

        elif k == ord('e'):
            exit()

    def __regular_iteration__(self):
        # get current frame and ends the processing when no more frames are detected
        ret, frame = self.cap.read()
        if not ret: 
            return False, '[INFO] All frames were processed.'
        
        # correct spherical distortion
        frame = self.__undistort__(frame)

        # ROI Arrays of tracking objects
        roi_array = []

        # Track every object and save past and current ROIs
        for otrack in self.object_trackers:
            roi_array.append(otrack.cXY)
            otrack.track(frame)
            roi_array.append(otrack.cXY)
 
        
        ret, message = self.camera_tracker.track(self.prev_frame, frame, roi_array)
        frame_id = self.iteration_counter + self.first_frame

        if not ret:
            return False, '[Error] {} (Frame {})'.format(message, frame_id)
       

        # display the full image with the ant in blue (TODO: Refactor this to make more general)
        tools.show_frame(frame, self.object_trackers[0].cXY, self.camera_tracker.roi, self.camera_tracker.features, frame_id)

        # save current frame and ROI center as previous for next iteration
        self.prev_frame = frame.copy()


        # Call the keyboard controller to handle key interruptions
        self.keyboard_controller()

        # Increase the iteration counter
        self.iteration_counter += 1

        return True, '[INFO] Frame {} was processed'.format(frame_id)

    def __release_cap__(self):
        self.cap.release()
        cv2.destroyAllWindows()


    def __export_data__(self):
        # TODO: This function needs to be rewritten to be able to handle more than 1 object tracker
        # and also to provide more general purpose semantic information
        last_frame = self.first_frame + self.iteration_counter
        percent_video = 100 * (self.first_frame + self.iteration_counter) / self.frame_count
        minutes_video = last_frame / self.fps / 60

        data = {
            'fps': self.fps,
            'first_frame': self.first_frame,
            'last_frame': last_frame,
            'percent': percent_video,
            'r_ac' : self.object_trackers[0].history,
            'affine_params': self.camera_tracker.history,
            'mse': self.camera_tracker.mse
        }
        self.__save_data__(data, minutes_video, percent_video)


    def __save_data__(self, data, minutes=None, percent=None):
        if not (minutes or percent):
            progress = ''
        else:
            summary = f'{minutes:.1f}min' if minutes else ''
            summary += '-' if (minutes and percent) else ''
            summary += f'{percent:.1f}%' if percent else ''
            progress = '_[{}]'.format(summary)

        data_file_dir = '{}{}.json'.format(self.video_path[:-4], progress)

        with open(data_file_dir, 'w') as json_file:
            json.dump(data, json_file)

    def track(self, video_path, start_in_frame=0):
        self.__digest_video_path(video_path)
        
        if self.iteration_counter == 0:
            retval, message = self.__first_iteration__(start_in_frame)
            if not retval:
                return retval, message

        while self.enabled:
            retval, message = self.__regular_iteration__()
            if not retval:
                break

        if message == '[INFO] All frames were processed.':
            retval = True

        self.__release_cap__()
        self.__export_data__()
        return retval, message

