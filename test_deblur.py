from deblur import AngleEstimator

#a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170536_12218', 625, 1000)
#a.estimate()


a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', 900, 1100)
a.estimate(debug=True)
def run():
    for i in range(10):
        offset = 1120
        a = AngleEstimator('/home/smorad/spin_odom/images/speer_data/20190130_170303_4856', offset+i, offset+i+1)
        a.estimate(debug=True)
    



#run()
