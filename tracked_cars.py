class Tracked_Cars():
    '''
    Keeps track of the car objects that were tracked based on their centroid
    '''
    def __init__(self, carId, centroid, x1, x2, y1, y2, maxDisappearedFrames=200):
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
        #get the corner coordinates of each tracked bounding box
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def setX1(self, x1):
        self.x1 = x1

    def setX2(self, x2):
        self.x2 = x2

    def setY1(self, y1):
        self.y1 = y1

    def setY2(self, y2):
        self.y2 = y2

    def getX1(self):
        return self.x1

    def getX2(self):
        return self.x2

    def getY1(self):
        return self.y1

    def getY2(self):
        return self.y2

    def setTracked(self, tracked):
        self.tracked = tracked
    
    def getArea(self):
        return (abs(self.x1-self.x2) * abs(self.y1-self.y2))

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