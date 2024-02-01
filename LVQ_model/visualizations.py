import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,  ConfusionMatrixDisplay
class Visualizations:
    def scatter_plot(self, train_data, prototype_data, epoch):
   
        diagnosis_labels = ["HC", "AD", "PD"]
        diagnosis_color = {"HC": "gray", "AD": "Blue", "PD": "violet"}
        prototype_color = {"HC": "black", "AD": "navy", "PD": "purple"}
        for diagnosis in  diagnosis_labels:
                
                x = train_data[diagnosis].values[:, 2:3]
                y = train_data[diagnosis].values[:, 3:4]
                
                p_x = prototype_data[diagnosis][0][0]
                p_y = prototype_data[diagnosis][1][0]
        
                plt.scatter(x, y, c=diagnosis_color[diagnosis], label=diagnosis)
                plt.scatter(p_x, p_y, c=prototype_color[diagnosis], label=diagnosis, s=100) 

        plt.legend()
        plt.savefig(f"scatter_{epoch}.png")
        plt.close()

    def bar_chart(self, data, epoch):
        courses = list(data.keys())
        values = list(data.values())
        # print(courses)
        fig = plt.figure(figsize = (10, 5))
        
        # creating the bar plot
        plt.bar(courses, values, color ='maroon', 
                width = 0.4)
        
        plt.xlabel("Predicted")
        plt.ylabel("Amount")
        plt.title("Predictions using LVQ with CCRE")
        plt.savefig(f"bar_chart_{epoch}.png")
        plt.close()
    
    def confussion_matrix(self, results_matrix, epoch):
        # print("Ik kwam hier confussion matrix")
        # print(results_matrix)
        actual = [i[0] for i in results_matrix]
        predicted = [i[2] for i in results_matrix]
        cm = confusion_matrix(actual, predicted, labels=["HC", "PD", "AD"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HC", "PD", "AD"])
        disp.plot()
        plt.savefig(f'confusion_matrix_{epoch}.png')
        plt.close()
    
