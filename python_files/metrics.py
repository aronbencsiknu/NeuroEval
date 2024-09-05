class Metrics():
    def __init__(self, ):
       
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    # return the predicted class from spike trains
    def return_predicted(self, output):
        
        _, predicted = output.sum(dim=0).max(dim=1)

        return predicted
   
    # calculate the confusion matrix
    def perf_measure(self, y_actual, y_hat):

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                self.TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                self.FP += 1
            if y_actual[i]==y_hat[i]==0:
                self.TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                self.FN += 1
        
    # calculate the precision
    def precision(self):
        return self.TP/(self.TP+self.FP+1e-8)

    # calculate the recall
    def recall(self):
        return self.TP/(self.TP+self.FN+1e-8)
    
    # calculate the harmonic mean of precision and recall
    def f1_score(self):
        return 2*(self.precision()*self.recall())/(self.precision()+self.recall()+1e-8)
    
    def get_scores(self):
        return self.TP, self.TN, self.FP, self.FN
        