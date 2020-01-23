import sys
import random

class PriorSampling:
	CPTs={}

	def __init__(self, netID):
		self.initialiseNet(netID)		

	def initialiseNet(self, netID):
		if netID == "burglary":
			self.CPTs["B"]={"+b":0.001, "-b":0.999}
			self.CPTs["E"]={"+e":0.002, "-e":0.998}
			self.CPTs["A"]={"+a|+b+e":0.95, "-a|+b+e":0.05, 
					"+a|+b-e":0.94, "-a|+b-e":0.06,
					"+a|-b+e":0.29, "-a|-b+e":0.71,
					"+a|-b-e":0.001, "-a|-b-e":0.999}
			self.CPTs["J"]={"+j|+a":0.90, "-j|+a":0.10, 
					"+j|-a":0.05, "-j|-a":0.95}
			self.CPTs["M"]={"+m|+a":0.70, "-m|+a":0.30, 
					"+m|-a":0.01, "-m|-a":0.99}
			self.CPTs["order"]=["B", "E", "A", "J", "M"]
			self.CPTs["parents"]={"B":None, "E":None, "A":"B,E", "J":"A", "M":"A"}

		elif netID == "sprinkler":
			self.CPTs["C"]={"+c":0.50, "-c":0.50}
			self.CPTs["S"]={"+s|+c":0.10, "-s|+c":0.90, 
					"+s|-c":0.50, "-s|-c":0.50}
			self.CPTs["R"]={"+r|+c":0.80, "-r|+c":0.20, 
					"+r|-c":0.20, "-r|-c":0.80}
			self.CPTs["W"]={"+w|+s+r":0.99, "-w|+s+r":0.01, 
					"+w|+s-r":0.90, "-w|+s-r":0.10,
					"+w|-s+r":0.90, "-w|-s+r":0.10,
					"+w|-s-r":0.00, "-w|-s-r":1.00}
			self.CPTs["order"]=["C", "S", "R", "W"]
			self.CPTs["parents"]={"C":None, "S":"C", "R":"C", "W":"S,R"}

		else:
			print("UNKNOWN network="+str(netID))
			exit(0)

	def sampleVariable(self, CPT, conditional):
		sampledValue=None
		randnumber=random.random()

		value1=CPT["+"+conditional]
		#value2=CPT["-"+conditional]

		if randnumber<=value1:
			sampledValue="+"+conditional
		else:
			sampledValue="-"+conditional

		return sampledValue.split("|")[0]

	def sampleVariables(self, printEvent):
		event=[]
		sampledVars={}

		for variable in self.CPTs["order"]:
			evidence=""
			conditional=""
			parents=self.CPTs["parents"][variable]
			if parents==None:
				conditional=variable.lower()
			else:
				for parent in parents.split(","):
					evidence+=sampledVars[parent]
				conditional=variable.lower()+"|"+evidence

			sampledValue=self.sampleVariable(self.CPTs[variable], conditional)
			event.append(sampledValue)
			sampledVars[variable]=sampledValue
				
		if printEvent: print(event)
		return event

class RejectionSampling:
    #These are class variables, shared among all objects
    #variable for holding a prior sampling class's object
    ps = None
    #Counter for samples where query Variable is true(after rejecting inconsistent samples)
    positive_outcomes = 0
    #otherwise counter
    negative_outcomes = 0
    
    
    def __init__(self, bn):
        #create object from sampling class
        self.ps=PriorSampling(bn)
    
    #this function is to return true if the sample of network is consistent with evidence
    #basically checks if values of evidence variables are same in query and sample
    def isConsistentWithEvidence(self,  evidence,  event):
        #If there is no evidence to consider
        if evidence == "": 
            return True
     
        #This splits the string in strings at "," character
        for i in evidence.split(","):
            #if that string is not in event sampled, return false
            if i not in event:
                return False
                
        #If all tests passsed, return true
	#print("Consistent!")
        return True
    
    def getIndexOfVariableInEventList(self,  queryVariable):
        #Load order from the net definition
        variables = self.ps.CPTs["order"]
        #Look for the query variable in the order list
        for i in range(0,  len(variables)):
            if variables[i] == queryVariable:
                return i 
        
        #No idea why?
        return None
                
    
    def singleEventGenerationAndEvaluation(self, evidence,  X):
        #This samples the network  and creates the event
        event = self.ps.sampleVariables(False)
        #proceed only when event is consistant with evidence otherwise sample is being rejected
        if self.isConsistentWithEvidence(evidence,  event):
            #Now its time to figure out if query variable is true or false in given sample
            #first find the index of the variable in event passed from prior sampling
            index = self.getIndexOfVariableInEventList(X)
            if ('+' in event[index]):
                print(self.positive_outcomes)
                self.positive_outcomes +=1
            else:
                self.negative_outcomes +=1
           
    #This function repeats the treatment of a single sample numberOfSamples times
    def eventLoop(self,  numberOfSamples, evidence,  X):
        for i in range(1,  numberOfSamples):
            self.singleEventGenerationAndEvaluation(evidence,  X)
            
    def normalize(self):
        #sum of all not rejected events
        sum =float( self.negative_outcomes + self.positive_outcomes)
        distibution = [float(self.positive_outcomes)/sum, float(self.negative_outcomes)/sum]
        return distibution
        

    def printProbabilityDistribution(self, numberOfSamples,  evidence,  X):
        #runs the dampling and evaluation
        self.eventLoop(numberOfSamples,  evidence,  X)
        distribution = self.normalize()
        print("P("+X+"|"+evidence+") = "), 
        print(distribution)

if __name__ == "__main__":
    #X is the query variable, for example for Burglray net use one of B E A J M
    X = str(sys.argv[1])
    #evidence observed, they are treated as a known information, so for example b=1, c=1, b=0 and c=1
    #these would be inputed as +b, +c, -b,+c
    evidence = str(sys.argv[2])
    #bn is a bayes network used by sampling class prior sampling
    #possible values are "burglary" and "sprinkler"
    bn = str(sys.argv[3])
    N = int(sys.argv[4])
    RJ = RejectionSampling(bn)
    RJ.printProbabilityDistribution(N, evidence,  X)


    
    
    
    
