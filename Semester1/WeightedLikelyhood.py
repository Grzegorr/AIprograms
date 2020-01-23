import sys
import random

class WeightedSampling:
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

    def oneWeightedSample(self,  e,  X):
        event = []
        w = 1
        #Run a loop and either sample or change a weigth
        for i in self.CPTs["order"]:
            #print(event)
            #Check if the variable is in evidence string
            #n = self.getIndexOfVariableInEventList(i)
            if i.lower() in e:
                #print("Weight reducing branch")
                #if so set the value in the event and reduce the weight
                #get value of variable from evidence
                bool = self.trueOrFalse(i,  e)
                #set this value to the event
                event.append(bool)
                #adjust weight accordingly
                w = w * self.returnVariableProbability(bool,  i,  event)
            else:
                #sample the variable from conditional probability and add to event
                logicValue = self.sample(self.returnVariableProbability(True, i,  event))
                event.append(logicValue)
        #Is the value of query variable true in the event?
        queryBool = self.isQueryTrue(X,  event)
        return [event,  w, queryBool]
    
    def isQueryTrue(self, X,  event):
        #get index of query variable in the event array
        index = self.getIndexOfVariableInEventList(X)
        return event[index]
    
    #functions checks if the variable in the event was true or false
    def trueOrFalse(self,  variable,  e):
        trueString = "+"+variable.lower()
        falseString = "-"+variable.lower()
        if trueString in e:
            return True
        if falseString in e:
            return False
        return None
    
    #This function knowing event part sampled so far gets a chance that the variable is bool
    def returnVariableProbability(self, bool,  var, event):
        varLower = var.lower()
        #if variable has no parents, return nonconditional probability
        parents=self.CPTs["parents"][var]
        if parents == None:
            #print("No Parents")
            return self.CPTs[var]["+"+varLower]
        
        #if has parents fetch value of parents from event so far and get conditional probability
        askString = "+"+varLower+"|"
        #run loop for all parents
        for parent in parents.split(","):
            #print(parent)
            #get index in event matrix for this parent
            n = self.getIndexOfVariableInEventList(parent)
            #print(n)
            #modify ask string depending on the current event true/false values
            if event[n] == True:
                askString = askString + "+"+parent.lower()
            else:
                askString = askString +"-"+parent.lower()
        
        #return the conditional probablity based on the knowledge from before the sample
        if bool == True:
            return self.CPTs[var][askString]
        else:
            return 1 - self.CPTs[var][askString]
     
  
    
    def getIndexOfVariableInEventList(self,  queryVariable):
        #Load order from the net definition
        variables = self.CPTs["order"]
        #Look for the query variable in the order list
        for i in range(0,  len(variables)):
            #print(i)
            if variables[i] == queryVariable:
                return i 
    
    #This function takes probability that event is true and returns TRue or False
    def sample(self, prob):
        #get random number
        randnumber=random.random()
        if randnumber > prob:
            return False
        else:
            return True

class WeightedLikelyhood:
    #These are class variables, shared among all objects
    #variable for holding a prior sampling class's object
    ws = None
    #Counter for samples where query Variable is true(after rejecting inconsistent samples)
    positive_outcomes = 0.0
    #otherwise counter
    negative_outcomes = 0.0
    
    def __init__(self, bn):
        #create object from sampling class
        self.ws=WeightedSampling(bn)
    
    #This function repeats the creation of an event with weight and assigns weights to positive_outcomes and negative outcomes
    def eventLoop(self,  numberOfSamples, evidence,  X):
        for i in range(1,  numberOfSamples):
            #generate event
            weightedSample = self.ws.oneWeightedSample(evidence, X)
            #print(weightedSample)
            #add weight as a positive or negative outcome
            if weightedSample[2] == True:
                self.positive_outcomes += weightedSample[1]
            if weightedSample[2] == False:
                self.negative_outcomes += weightedSample[1]
            
    def normalize(self):
        #sum of all not rejected events
        sum = float( self.negative_outcomes + self.positive_outcomes)
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
    #N is number of samples to be taken for probability estmation
    N = int(sys.argv[4])
    #
    WL = WeightedLikelyhood(bn)
    WL.printProbabilityDistribution(N, evidence,  X)


    
    
    
    
