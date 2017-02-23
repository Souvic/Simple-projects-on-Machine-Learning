import theano
from theano import tensor as T
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.utils import np_utils

def init():
    board=np.ones((3,3))/2
    return board
def play(board,x):
    loc=np.where(board==0.5)
    n=np.random.randint(len(loc[1]), size=10)
    n=n[1]
    board[loc[0][n],loc[1][n]]=x
    return board

def state(board):
    for i in range(0,3):
        if(board[i,1]==board[i,0]==board[i,2]==1):
            return 1
        if(board[1,i]==board[0,i]==board[2,i]==1):
            return 1
        if(board[i,1]==board[i,0]==board[i,2]==0):
            return 0
        if(board[1,i]==board[0,i]==board[2,i]==0):
            return 0
    if(board[1,1]==board[0,0]==board[2,2]==0):
        return 0
    if(board[1,1]==board[0,0]==board[2,2]==1):
        return 1
    if(board[2,0]==board[0,0]==board[0,2]==1):
        return 1
    if(board[2,0]==board[0,0]==board[0,2]==0):
        return 0

    return 0.5


def yo(i):
    b=()
    iii=np.random.randint(2,size=i)
    for ii in range(0,i):
        board=init()
        x=iii[ii]
        t=0

        while((state(board)==0.5)&(t<9)):
            board=play(board,x)
            #print(board)
            #print " "
            x=1-x
            t=t+1
            b=np.append(b,board.flatten())
            b=np.append(b,[5,6])
        b[np.where(b==5)]=(state(board)==0)*1
        b[np.where(b==6)]=(state(board)==1)*1

    return(b.reshape(len(b)/11,11))



    

       
        


model = Sequential()
model.add(Dense(output_dim=64, input_dim=9, init='normal', activation='relu'))
model.add(Dense(output_dim=64, init='normal', activation='relu'))
model.add(Dense(output_dim=64, init='normal', activation='relu'))
model.add(Dense(output_dim=2, init='normal', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])




for iter in range(500):
        data=yo(1000)
        train_X=data[0:700,0:9]
        #print train_X.shape
        test_X= data[700:1000,0:9]
        train_y=data[0:700,9:11]
        test_y =data[700:1000,9:11]
        
        model.train_on_batch(train_X, train_y)
        loss_and_metrics = model.evaluate(test_X, test_y, batch_size=300)
        print loss_and_metrics
