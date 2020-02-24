class Tracked_Cars():
    '''
    Keeps track of the car objects that were tracked based on their centroid
    '''
    def __init__(self, carId, centroid, maxDisappearedFrames=40):
        '''
        initialize the general object to be found
        '''
        #placeholder for the next id of the tracked car
        self.carId = carId
        #to indicate if this car was tracked or not
        self.tracked = True
        #to track how many frames we have lost the object
        self.disappearedFrames = 0
        #define how many frames must have passed to delete the object from the tracker
        self.maxDisappearedFrames = maxDisappearedFrames
        #define the centroid of the car
        self.centroid = centroid 


    def setTracked(self, tracked):
        self.tracked = tracked
    
    def getTracked(self):
        return self.tracked
    
    def setCentroid(self, centroid):
        self.centroid = centroid
    
    def getCentroid(self):
        return self.centroid

    def setDisappearedFames(self, disappearedFrames):
        self.disappearedFrames = disappearedFrames
    
    def getDisappearedFrames(self):
        return self.disappearedFrames

    def setCarId(self,carId):
        self.carId = carId

    def getCarId(self):
        return self.carId

    def toString(self):
        s = "Id:" + str(self.carId)
        return s