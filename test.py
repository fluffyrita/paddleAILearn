
def LCS(strA,strB):
    lenA = len(strA)
    lenB = len(strB)    
    N = [[0]*(lenB+1) for i in range(lenA+1)]
    H = [[0]*(lenB+1) for i in range(lenA+1)]
    for i in range(1,lenA+1):
        for j in range(1,lenB+1):
            if strA[i-1] == strB[j-1]:
                N[i][j] = N[i-1][j-1]+1
                H[i][j] = 'leftTop'
            elif strA[i-1] !=strB[j-1] and N[i][j-1] >= N[i-1][j]:
                N[i][j] = N[i][j-1]
                H[i][j] = 'left'
            else:
                N[i][j] = N[i-1][j]
                H[i][j] = 'top'
    print(N)
    print(H)

def main():
    strA = 'ABCBDAB'
    strB = 'BDCABA'
    print(LCS(strA,strB))    

if __name__=='__main__':
    main()
