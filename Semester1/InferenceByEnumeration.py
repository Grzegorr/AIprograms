import sys
import csv

class Network:
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

        #setting network to "desease" triggers assignment 1A and will prompt user for information
        elif netID == "disease":
            #Prompting information and computing probailities
            networkGen = DiseaseNetworkSetup()
            [dT,  dF,  tTgT, tFgT, tTgF, tFgF] = networkGen.returnVariables(True)
            #Setting up actual network
            self.CPTs["D"] = {"+d":dT, "-d":dF}
            self.CPTs["T"] = {"+t|+d":tTgT, "-t|+d":tFgT, "+t|-d":tTgF,  "-t|-d":tFgF}
            self.CPTs["order"] = ["D",  "T"]
            self.CPTs["parents"] = {"D":None,  "T":"D"}
            
        elif netID == "smoking":
            netGenerator = SmokingNetworkSetup()
            print("Not yet done xd")
            [aT, aF,  ppT, ppF, gT,  gF, bedT, bedF, alT, alF, sTgTT, sFgTT, sTgFT, sFgFT, sTgFF, sFgFF, sTgTF, sFgTF, yfTgT,  yfFgT, yfTgF, yfFgF, lcTgTT, lcFgTT, lcTgFT, lcFgFT, lcTgTF, lcFgTF, lcTgFF, lcFgFF, adTgT,  adFgT,  adTgF, adFgF, cTgTT, cFgTT, cTgFT, cFgFT, cTgFF, cFgFF, cTgTF, cFgTF, fTgTT, fFgTT, fTgFT, fFgFT, fTgFF, fFgFF, fTgTF, fFgTF, caTgTT, caFgTT, caTgFT, caFgFT, caTgFF, caFgFF, caTgTF, caFgTF] = netGenerator.generateProbabilities()
            #Actual network setup
            self.CPTs["A"] = {"+a":aT, "-a":aF}
            self.CPTs["PP"] = {"+pp":ppT, "-pp":ppF}
            self.CPTs["G"] = {"+g":gT, "-g":gF}
            self.CPTs["BED"] = {"+bed":bedT, "-bed":bedF}
            self.CPTs["AL"] = {"+al":alT, "-al":alF}
            self.CPTs["S"] = {"+s|+a+pp":sTgTT, "-s|+a+pp":sFgTT, "+s|-a+pp":sTgFT, "-s|-a+pp":sFgFT, "+s|-a-pp":sTgFF, "-s|-a-pp":sFgFF, "+s|+a-pp":sTgTF, "-s|+a-pp":sFgTF}
            self.CPTs["YF"] = {"+yf|+s":yfTgT,  "-yf|+s":yfFgT, "+yf|-s":yfTgF, "-yf|-s":yfFgF}
            self.CPTs["LC"] = {"+lc|+s+g":lcTgTT, "-lc|+s+g":lcFgTT, "+lc|-s+g":lcTgFT, "-lc|-s+g":lcFgFT, "+lc|+s-g":lcTgTF, "-lc|+s-g":lcFgTF, "+lc|-s-g":lcTgFF, "-lc|-s-g":lcFgFF}
            self.CPTs["AD"] = {"+ad|+g":adTgT,  "-ad|+g":adFgT,  "+ad|-g":adTgF, "-ad|-g":adFgF}
            self.CPTs["C"] = {"+c|+al+lc":cTgTT, "-c|+al+lc":cFgTT, "+c|-al+lc":cTgFT, "-c|-al+lc":cFgFT, "+c|-al-lc":cTgFF, "-c|-al-lc":cFgFF, "+c|+al-lc":cTgTF, "-c|+al-lc":cFgTF}
            self.CPTs["F"] = {"+f|+lc+c":fTgTT, "-f|+lc+c":fFgTT, "+f|-lc+c":fTgFT, "-f|-lc+c":fFgFT, "+f|-lc-c":fTgFF, "-f|-lc-c":fFgFF, "+f|+lc-c":fTgTF, "-f|+lc-c":fFgTF}
            self.CPTs["CA"] = {"+ca|+ad+f":caTgTT, "-ca|+ad+f":caFgTT, "+ca|-ad+f":caTgFT, "-ca|-ad+f":caFgFT, "+ca|-ad-f":caTgFF, "-ca|-ad-f":caFgFF, "+ca|+ad-f":caTgTF, "-ca|+ad-f":caFgTF}
            self.CPTs["order"] = ["A", "PP", "G", "BED", "AL", "S", "YF", "LC", "AD", "C", "F", "CA"]
            self.CPTs["parents"] = {"A":None, "PP":None, "G":None, "BED":None, "AL":None, "S":"A,PP", "YF":"S", "LC":"S,G", "AD":"G", "C":"AL,LC", "F":"LC,C", "CA":"AD,F"}
            

        else:
            print("UNKNOWN network="+str(netID))
            exit(0)


class DiseaseNetworkSetup:
    dT = 0.0
    dF = 0.0
    tTgT = 0.0
    tFgT = 0.0
    tFgT = 0.0
    tFgF = 0.0
    
    def __init__(self):
        #Prompting for probability of desease
        print("Please type in probability of desease in the population.")
        self.dT = input()
        self.dF = 1.0 - self.dT
        #First prompt for quality of the test
        print("Please type in probability that the test is positive given the person has the disease.")
        self.tTgT = input()
        self.tFgT = 1 - self.tTgT
        print("Please type probability that the test is negative given the person does not have the disease.")
        self.tFgF = input()
        self.tTgF = 1 - self.tFgF
        
    def returnVariables(self,  ifPrint):
        if ifPrint == True:
            print([self.dT,  self.dF,  self.tTgT, self.tFgT, self.tTgF, self.tFgF])
        return [self.dT,  self.dF,  self.tTgT, self.tFgT, self.tTgT, self.tFgF]
        
class SmokingNetworkSetup:
    #Preprogrammed order for the variables IN CSV NOT NETWORK
    order_in_csv = ["S",  "YF",  "A",  "PP",  "G",  "AD",  "BED",  "CA",  "F",  "AL",  "C", "LC"]
    events = []
    purged_events = []
    #Init, taking care of reading the file
    def __init__(self):
        #CSV reading
        with open('assignmentDataset.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_count = 0
            #converting all rows exept first into events
            for row in csv_reader:
                if row_count != 0:
                    self.events.append(row)
                    #print(self.events[row_count-1])
                    row_count += 1
                else:
                    row_count += 1
        print(str(row_count-1) + " events was read in.")


    def generateProbabilities(self):
        #generate probabilities for node A
        [aT,  aF] = self.probabilityDistribution("+a", "")
        #generate probabilities for node PP
        [ppT,  ppF] = self.probabilityDistribution("+pp", "")
        #generate probabilities for node G
        [gT,  gF] = self.probabilityDistribution("+g", "")
        #generate probabilities for node BED
        [bedT,  bedF] = self.probabilityDistribution("+bed", "")
        #generate probabilities for node AL
        [alT,  alF] = self.probabilityDistribution("+al", "")
        #generate probabilities for node S (Smoking)
        [sTgTT, sFgTT] = self.probabilityDistribution("+s", "+a,+pp")
        [sTgFT, sFgFT] = self.probabilityDistribution("+s", "-a,+pp")
        [sTgFF, sFgFF] = self.probabilityDistribution("+s", "-a,-pp")
        [sTgTF, sFgTF] = self.probabilityDistribution("+s", "+a,-pp")
        #generate probabilities for node YF (Yellow Fingers)
        [yfTgT, yfFgT] = self.probabilityDistribution("+yf", "+s")
        [yfTgF, yfFgF] = self.probabilityDistribution("+yf", "-s")
        #generate probabilities for node LC (Lung Cancer)
        [lcTgTT, lcFgTT] = self.probabilityDistribution("+lc", "+s,+g")
        [lcTgFT, lcFgFT] = self.probabilityDistribution("+lc", "-s,+g")
        [lcTgFF, lcFgFF] = self.probabilityDistribution("+lc", "-s,-g")
        [lcTgTF, lcFgTF] = self.probabilityDistribution("+lc", "+s,-g")
        #generate probabilities for node AD (AttentionDisorder)
        [adTgT, adFgT] = self.probabilityDistribution("+ad", "+g")
        [adTgF, adFgF] = self.probabilityDistribution("+ad", "+g")
        #generate probabilities for node C (caughing)
        [cTgTT, cFgTT] = self.probabilityDistribution("+c", "+al,+lc")
        [cTgFT, cFgFT] = self.probabilityDistribution("+c", "-al,+lc")
        [cTgFF, cFgFF] = self.probabilityDistribution("+c", "-al,-lc")
        [cTgTF, cFgTF] = self.probabilityDistribution("+c", "+al,-lc")
        #generate probabilities for node F (fatigue)
        [fTgTT, fFgTT] = self.probabilityDistribution("+f", "+c,+lc")
        [fTgFT, fFgFT] = self.probabilityDistribution("+f", "-c,+lc")
        [fTgFF, fFgFF] = self.probabilityDistribution("+f", "-c,-lc")
        [fTgTF, fFgTF] = self.probabilityDistribution("+f", "-c,-lc")
        #generate probabilities for node CA (Car Accident)
        [caTgTT, caFgTT] = self.probabilityDistribution("+ca", "+ad,+f")
        [caTgFT, caFgFT] = self.probabilityDistribution("+ca", "-ad,+f")
        [caTgFF, caFgFF] = self.probabilityDistribution("+ca", "-ad,-f")
        [caTgTF, caFgTF] = self.probabilityDistribution("+ca", "+ad,-f")
        
        
        return [aT, aF,  ppT, ppF, gT,  gF, bedT, bedF, alT, alF, sTgTT, sFgTT, sTgFT, sFgFT, sTgFF, sFgFF, sTgTF, sFgTF, yfTgT,  yfFgT, yfTgF, yfFgF, lcTgTT, lcFgTT, lcTgFT, lcFgFT, lcTgTF, lcFgTF, lcTgFF, lcFgFF, adTgT,  adFgT,  adTgF, adFgF, cTgTT, cFgTT, cTgFT, cFgFT, cTgFF, cFgFF, cTgTF, cFgTF, fTgTT, fFgTT, fTgFT, fFgFT, fTgFF, fFgFF, fTgTF, fFgTF, caTgTT, caFgTT, caTgFT, caFgFT, caTgFF, caFgFF, caTgTF, caFgTF]
        #return [aT, aF, ppT, ppF, gT, gF, bedT, bedF, alT, alF, sTgTT, sFgTT, sTgFT, sFgFT, sTgFF, sFgFF, sTgTF, sFgTF,  yfTgT,  yfFgT, yfTgF, yfFgF, lcTgTT, lcFgTT, lcTgFT, lcFgFT, lcTgTF, lcFgTF, lcTgFF, lcFgFF, adTgT,  adFgT, adTgF, adFgF, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def probabilityDistribution(self, variable, evidence):
        pT = (self.countEvents(evidence+","+variable)+1)/(self.countEvents(evidence) + 2)
        return [pT,  1-pT]
        
    #This function preselects events so only those meeting the observation remain
    #format here is like +b,+m
    def countEvents(self, conditions):
        events_count = 0
        fail = 0
        #clean the purged events variable
        self.purged_events = []
        for e in self.events:
            #reset fail flag
            fail = 0
            #do chceck only if conditions actually exist
            #print("Conditions are:" + conditions + ".")
            if len(conditions) != 0:
                for c in conditions.split(","):
                    #do this only when c is not None, this happens when computing nodes without parents
                    if len(c) != 0:
                        #was the condition 1 or 0?
                        value = 4
                        if "+" in c:
                            value = 1
                        if "-" in c:
                            value = 0
                        #drop + or - from the variable
                        c = c[1:len(c)]
                        #Get index in order of csv
                        c = c.upper()
                        print(c)
                        index = self.getIndexCSV(c)
                        if int(e[index]) != int(value):
                            #this condition was not met, set the fail flag
                            fail = 1
            #copy event to purged_events
            if fail == 0:
                events_count += 1
        #print(str(events_count) + " events met conditions: " + conditions)
        return float(events_count)
                
    def getIndexCSV(self, var):
        counter = 0
        for c in self.order_in_csv:
            if var == self.order_in_csv[counter]:
                return counter
            counter += 1


class Enumeration:
    #Variable to store network to work on.
    net = None
    
    def __init__(self,  bn):
        self.net = Network(bn)
        
    #this function takes unnormalized probability distribution and prints out a normalized probability distribution    
    def printProbabilityDistribution(self, evidence, X):
        #get unnormalized probability distribution
        unnormalized_distribution = self.getNotNormalizedProbabilities(evidence,  X)
        #normalize
        sum = unnormalized_distribution[0] + unnormalized_distribution[1]
        normalized_distribution = [0, 0]
        normalized_distribution[0] = unnormalized_distribution[0]/sum
        normalized_distribution[1] = unnormalized_distribution[1]/sum
        print(normalized_distribution)
        
    def getNotNormalizedProbabilities(self,  evidence,  X):
        #Load variables from network
        variablesToEnumerate = self.net.CPTs["order"]
        #Extending evidence with the True and False versions of query variable
        ext_evidenceT = evidence + ",+" + X.lower()
        ext_evidenceF = evidence + ",-" + X.lower()
        ProbT = self.enumerate(variablesToEnumerate, ext_evidenceT)
        ProbF = self.enumerate(variablesToEnumerate, ext_evidenceF)
        return [ProbT,  ProbF]

    def enumerate(self, variables,  extended_evidence):
        #print("Enumerate started")
        #print(variables)
        if len(variables) == 0:
            #print("NO VARIABLES LEFT")
            return 1.0
        Y = variables[0]
        y = Y.lower()
        yT = "+" + y
        yF = "-" + y
        #print(y)
        #print(extended_evidence.split(","))
        if yT in extended_evidence.split(",") or yF in extended_evidence.split(","):
            print("<<<<<<<<<<<<<<<<<Variabl " + y + " found in evidence")
            return self.net.CPTs[Y][self.conditionalString(Y, extended_evidence,  False)] * self.enumerate(variables[1:len(variables)], extended_evidence)
        else: 
            #print("Sum node. Variable: " + y )
            #print(variables)
            #print(variables[1:len(variables)-1])
            #print("-------Parts of a multiplication")
            #print(self.net.CPTs[Y]["+" + y + self.conditionalString(Y, extended_evidence + ",+" + y,  True)])
            #print(self.enumerate(variables[1:len(variables)], extended_evidence + ",+" + y))
            sum1 = self.net.CPTs[Y]["+" + y + self.conditionalString(Y, extended_evidence + ",+" + y,  True)] * self.enumerate(variables[1:len(variables)], extended_evidence + ",+" + y)
            #print(sum1)
            sum2 = self.net.CPTs[Y]["-" + y + self.conditionalString(Y, extended_evidence + ",-" + y,  True)] * self.enumerate(variables[1:len(variables)], extended_evidence + ",-" + y)
            #print("Variable: " + Y +" Sums in Enumeration: ")
            #print(sum1)
            #print(sum2)
            return sum1 + sum2

    def conditionalString(self, Y, evidence,  partial):
        string = ""
        #do not run this if only partial string is needed (when arbitrary setting the dependent variable when summing)
        if partial == False:
            #first part of string
            string = self.valueInEvidence(Y, evidence)
        #if there is no parents, exit here
        if self.net.CPTs["parents"][Y] == None:
            return string
        string += "|"
        #print("Parents: " + self.net.CPTs["parents"][Y])
        for b in self.net.CPTs["parents"][Y].split(","):
            #print("Parent loop in conditional string generation")
            string += self.valueInEvidence(b,  evidence)
            #print("Updated string in parents loop: " + string)
        #print(string)
        return string
        

    def valueInEvidence(self, Y, evidence):
        #print(evidence)
        output_string = ""
        y = Y.lower()
        #print("valueInEvidence query variable: +"+ y)
        if ("+"+ y) in evidence.split(","):
            output_string = "+" + y
        if ("-"+ y) in evidence.split(","):
            output_string = "-" + y
        #print("value of evidence output string" + output_string)
        return output_string


if __name__ == "__main__":
    #X is the query variable, for example for Burglray net use one of B E A J M
    X = str(sys.argv[1])
    #evidence observed, they are treated as a known information, so for example b=1, c=1, b=0 and c=1
    #these would be inputed as +b, +c, -b,+c
    evidence = str(sys.argv[2])
    #bn is a bayes network used by sampling class prior sampling
    #possible values are "burglary", "sprinkler", "disease" , "smoking"
    bn = str(sys.argv[3])
    IbE = Enumeration(bn)
    IbE.printProbabilityDistribution(evidence,  X)
    
    
    
