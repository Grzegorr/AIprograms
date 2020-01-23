import sys

#This class stores a network and provides method to return probability values from this network
class Network:
    CPTs={}

    def __init__(self, netID):
        self.networkSetup(netID)		

    def networkSetup(self, netID):
        
        if netID == "weather":
            #Hidden States Transitions, convention is that CPTs[ON][OFF] is probability of ON given last time step it was OFF
            self.CPTs["ON"]={"ON":0.7, "OFF":0.3}
            self.CPTs["OFF"]={"ON":0.3, "OFF":0.7}
            #Conditional probabilities of observing observable variables given the hidden state
            self.CPTs["HOT"]={"ON":0.4, "OFF":0.1}
            self.CPTs["WARM"]={"ON":0.4, "OFF":0.45}
            self.CPTs["COLD"]={"ON":0.2, "OFF":0.45}
        else:
            print("UNKNOWN network="+str(netID))
            exit(0)

    #returning a transition probability between two hidden states
    def returnHiddenProbability(self, now,  before):
        return self.CPTs[now][before]
    
    #returns probability of a given observation based on current hidden state
    def returnObservationProbability(self, observation,  hiddenState):
        return self.CPTs[observation][hiddenState]
        
#This class performs forward algorithm and prints out probability 
#of appearance of the query obseravation 
class HiddenMarkow:
    net = None
    observations = []
    alphaON = [0.5] 
    alphaOFF = [0.5]
    
    def __init__(self, inputString):
        #Read in network information
        self.net = Network("weather")
        self.observations = [x for x in inputString]
        #go from single letter to full words for observations
        #This is to allow the user to use single letters for faster chanes of queries
        for i in range(0,  len(self.observations)):
            if self.observations[i] == "H":
                self.observations[i] = "HOT"
                
            if self.observations[i] == "C":
                self.observations[i] = "COLD"
            if self.observations[i] == "W":
                self.observations[i] = "WARM"
        for i in range(0,  len(self.observations)):
            self.forward(self.observations[i])
        #print(self.alphaON)
        #print(self.alphaOFF)
        self.printProbability()
     
    #Runs the forword algorithm without normalization
    #Weights of trellis are stored in self.alphaON/OFF by appending these arrays
    def forward(self, observationToday):
        previousAlphaON = self.alphaON[len(self.alphaON) - 1]
        previousAlphaOFF = self.alphaOFF[len(self.alphaOFF) - 1]
        #For taday being ON
        alphaON = self.net.returnObservationProbability(observationToday, "ON") * (self.net.returnHiddenProbability("ON", "ON") * previousAlphaON + self.net.returnHiddenProbability("ON", "OFF") * previousAlphaOFF)
        #for today being OFF
        alphaOFF = self.net.returnObservationProbability(observationToday, "OFF") * (self.net.returnHiddenProbability("OFF", "ON") * previousAlphaON + self.net.returnHiddenProbability("OFF", "OFF") * previousAlphaOFF)
        self.alphaON.append(alphaON)
        self.alphaOFF.append(alphaOFF)

    #Computes probability by adding weights of both trellis
    def printProbability(self):
        prob = self.alphaON[len(self.alphaON) - 1] + self.alphaOFF[len(self.alphaOFF) - 1]
        print("Probability of this sequence is " + str(prob))

#Code run on the start of the script
if __name__ == "__main__":
    #Input string containing the observation string in form like CHWHC
    inputString = str(sys.argv[1])
    HM = HiddenMarkow(inputString)





















