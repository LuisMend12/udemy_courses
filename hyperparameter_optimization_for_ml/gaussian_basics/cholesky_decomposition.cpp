// CPP program to find decompose of a
// matrix using Cholesky Decomposition
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

void choleskyDecomposition(vector<vector<int>> matrix){
    int n = matrix.size();

    // to store the lower triangular matrix
    vector<vector<int>> lower(n, vector<int>(n, 0));

    // Decomposing a matrix into Lower Triangular
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            int sum = 0;

            // summation for diagonals
            if (j == i) {
                for (int k = 0; k < j; k++)
                    sum += pow(lower[j][k], 2);
                lower[j][j] = sqrt(matrix[j][j] - sum);
            } else {

                // Evaluating L(i, j) using L(j, j)
                for (int k = 0; k < j; k++)
                    sum += (lower[i][k] * lower[j][k]);
                lower[i][j] = (matrix[i][j] - sum) / lower[j][j];
            }
        }
    }

    // Displaying Lower Triangular Matrix
    for (int i = 0; i < n; i++) {
        
        // Lower Triangular
        for (int j = 0; j < n; j++)
            cout << lower[i][j] << " ";
        cout << endl;
    }

    cout<<endl;

    // Displaying Transpose of Lower Triangular Matrix
    for (int i = 0; i < n; i++) {
        
        // Lower Triangular
        for (int j = 0; j < n; j++)
            cout << lower[j][i] << " ";
        cout << endl;
    }
}

int main() {
    vector<vector<int>> matrix = { 
        { 4, 12, -16 },
        { 12, 37, -43 },
        { -16, -43, 98 } 
    };
    choleskyDecomposition(matrix);
    return 0;
}