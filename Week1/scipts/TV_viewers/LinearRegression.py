import pandas as pd
from sklearn import  linear_model

def get_data(file_name):
    data = pd.read_csv(file_name)
    flash_x_parameter = []
    flash_y_parameter = []
    arrow_x_parameter = []
    arrow_y_parameter = []
    for x1, y1, x2, y2 in zip(data['flash_episode_number'], data['flash_us_viewers'], data['arrow_episode_number'], data['arrow_us_viewers']):
        flash_x_parameter.append([float(x1)])
        flash_y_parameter.append([float(y1)])
        arrow_x_parameter.append([float(x2)])
        arrow_y_parameter.append([float(y2)])
    return flash_x_parameter, flash_y_parameter, arrow_x_parameter, arrow_y_parameter

# Function to know which TV show will have more viewers
def predict_more_viewers(x1, y1, x2, y2):
    # Create linear regression object
    regr1 = linear_model.LinearRegression()
    regr1.fit(x1, y1)
    predict_outcome1 = regr1.predict(9)
    print predict_outcome1
    regr2 = linear_model.LinearRegression()
    regr2.fit(x2, y2)
    predict_outcome2 = regr2.predict(9)
    print predict_outcome2
    if predict_outcome1 > predict_outcome2:
        print "The flash TV show will have more viewer for next week"
    else:
        print "The arrow TV show will have more viewer for next week"

x1, y1, x2, y2 = get_data('input_data.csv')
predict_more_viewers(x1, y1, x2, y2)