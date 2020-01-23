import sys

class SystemInitialization:
    CPTs={}

    def __init__(self, sysID):
        self.initialiseSystem(sysID)		

    def initialiseSystem(self, sysID):
        if sysID == "weather":
            self.CPTs["S"]={"S":0.8, "R":0.2, "F":0.2}
            self.CPTs["R"]={"S":0.05, "R":0.6, "F":0.3}
            self.CPTs["F"]={"S":0.15, "R":0.2, "F":0.5}
        else:
            print("UNKNOWN network="+str(sysID))
            exit(0)

    def returnConditionalProbability(self, Q,  E):
        return self.CPTs[Q][E]
        
class Computation:
    Sys=None
    
    def __init__(self,  ):
        self.Sys = SystemInitialization("weather")
    
    def printProbability(self,  A,  B, C, D):
        p1 = self.Sys.returnConditionalProbability(A, B)
        p2 = self.Sys.returnConditionalProbability(B, C)
        prob = p1 * p2
        print(prob)
        
if __name__ == "__main__":
    WaTmr1 = str(sys.argv[1])
    WTmr = str(sys.argv[2])
    WbTmr1 = str(sys.argv[3])#aka waether today
    WbTmr2 = str(sys.argv[4])
    
    Cmp = Computation()
    Cmp.printProbability(WaTmr1, WTmr,  WbTmr1,  WbTmr2)
