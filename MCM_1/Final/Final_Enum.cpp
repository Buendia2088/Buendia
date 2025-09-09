#include<iostream>
#include<fstream>
#include<vector>
#include<cstring>
using namespace std;

ifstream fin("res_dicreate_backup.csv");
ofstream fout("result5.csv");
void Split(string str, vector<string>& arr, char ch=',')
{
    string word = "";
    for (auto x : str)
    {
        if (x == ch)
        {
            arr.push_back(word);
            word = "";
        }
        else
        {
            word = word + x;
        }
    }
    arr.push_back(word);
}
double MinMaxScaler(double x, double min, double max, double new_min, double new_max) {
    return (x - min) / (max - min) * (new_max - new_min) + new_min;
}
int main()
{
    string line;
    vector<string> row;
    vector<vector<string>> data;
    getline(fin, line); // skip the first line
    double min_val = 1e9, max_val = -1e9;
    while (getline(fin, line))
    {
        row.clear();
        Split(line, row);
        double val = stod(row[1]);
        min_val = min(min_val, val);
        max_val = max(max_val, val);
        data.push_back(row);
    }

    // 对data[i][1]进行MinMax标准化并放大到（0，2）
    for(int i = 0; i < data.size(); i++)
    {
        double val = stod(data[i][1]);
        val = MinMaxScaler(val, min_val, max_val, 0, 100);
        data[i].push_back(to_string(val));
    }

    int result[3][2];
    double best = -1;
    double best_t1 = 0;
    double best_t2 = 0;
    double best__ = -1;
    double best_t1_ = 0;
    double best_t2_ = 0;
    double best_weight = 0;
    for(double Weight = 0.1; Weight <0.33; Weight+=0.1)
    {   
        cout<<Weight<<endl;
        for(double therhold1 = 20; therhold1 <= 60; therhold1 += 0.5)
        {
            for(double therhold2 = therhold1; therhold2 <= 70; therhold2 += 0.5)
            {
                memset(result, 0, sizeof(result));
                for(int i = 0; i < data.size(); i++)
                {
                    if(data[i][0] == "0")
                    {
                        if(stod(data[i][2]) < therhold1)
                        {
                            result[0][1]++;
                        }
                        else
                        {
                            result[0][0]++;
                        }
                    }
                    else if(data[i][0] == "50")
                    {
                        if(stod(data[i][2]) < therhold2)
                        {
                            result[1][1]++;
                        }
                        else
                        {
                            result[1][0]++;
                        }
                    }
                    else
                    {
                        if(stod(data[i][2]) >= therhold2)
                        {
                            result[2][1]++;
                        }
                        else
                        {
                            result[2][0]++;
                        }
                    }
                }
                double a1 = double (result[0][1]) / (result[0][0] + result[0][1]);
                double a2 = double (result[1][1]) / (result[1][0] + result[1][1]);
                double a3 = double (result[2][1]) / (result[2][0] + result[2][1]);
                double avg = Weight * a1 + (1-Weight) * 0.5 * a2 + (1-Weight) * 0.5 * a3;
                double sigma = Weight * (a1 - avg) * (a1 - avg) + (1-Weight) * 0.5 * (a2 - avg) * (a2 - avg) + (1-Weight) * 0.5 * (a3 - avg) * (a3 - avg);


                //cout << "Therhold1: " << therhold1 << " Therhold2: " << therhold2 << endl;
                fout << "Therhold1: " << therhold1 << " Therhold2: " << therhold2 << endl;
                fout << "0: " << a1 << endl;
                fout << "1: " << a2 << endl;
                fout << "2: " << a3 << endl;
                //cout <<endl;
                if(avg / sigma  > best)
                {
                    best = avg / sigma;
                    best_t1 = therhold1;
                    best_t2 = therhold2;
                }
            }
        }
        cout << "Best: " << best << " Therhold1: " << best_t1 << " Therhold2: " << best_t2 << endl;
        if(best > best__)
        {
            best__ = best;
            best_t1_ = best_t1;
            best_t2_ = best_t2;
            best_weight = Weight;
        }
    }
    
    return 0;
}

// #include<iostream>
// #include<fstream>
// #include<vector>
// #include<cstring>
// using namespace std;

// ifstream fin("res_dicreate_backup.csv");
// ofstream fout("result1.csv");
// void Split(string str, vector<string>& arr, char ch=',')
// {
//     string word = "";
//     for (auto x : str)
//     {
//         if (x == ch)
//         {
//             arr.push_back(word);
//             word = "";
//         }
//         else
//         {
//             word = word + x;
//         }
//     }
//     arr.push_back(word);
// }
// double MinMaxScaler(double x, double min, double max, double new_min, double new_max) {
//     return (x - min) / (max - min) * (new_max - new_min) + new_min;
// }
// double F1Score(int tp, int fp, int fn) {
//     double precision = double(tp) / (tp + fp);
//     double recall = double(tp) / (tp + fn);
//     return 2 * precision * recall / (precision + recall);
// }
// int main()
// {
//     string line;
//     vector<string> row;
//     vector<vector<string>> data;
//     getline(fin, line); // skip the first line
//     double min_val = 1e9, max_val = -1e9;
//     while (getline(fin, line))
//     {
//         row.clear();
//         Split(line, row);
//         double val = stod(row[1]);
//         min_val = min(min_val, val);
//         max_val = max(max_val, val);
//         data.push_back(row);
//     }

//     // 对data[i][1]进行MinMax标准化并放大到（0，100）
//     for(int i = 0; i < data.size(); i++)
//     {
//         double val = stod(data[i][1]);
//         val = MinMaxScaler(val, min_val, max_val, 0, 100);
//         data[i].push_back(to_string(val));
//     }

//     int result[3][2];
//     double best_f1 = -1;
//     double best_t1 = 0;
//     double best_t2 = 0;
//     for(double therhold1 = 0; therhold1 <= 100; therhold1 += 0.5)
//     {
//         for(double therhold2 = therhold1; therhold2 <= 100; therhold2 += 0.5)
//         {
//             memset(result, 0, sizeof(result));
//             for(int i = 0; i < data.size(); i++)
//             {
//                 if(data[i][0] == "0")
//                 {
//                     if(stod(data[i][2]) < therhold1)
//                     {
//                         result[0][1]++;
//                     }
//                     else
//                     {
//                         result[0][0]++;
//                     }
//                 }
//                 else if(data[i][0] == "50")
//                 {
//                     if(stod(data[i][2]) < therhold2)
//                     {
//                         result[1][1]++;
//                     }
//                     else
//                     {
//                         result[1][0]++;
//                     }
//                 }
//                 else
//                 {
//                     if(stod(data[i][2]) >= therhold2)
//                     {
//                         result[2][1]++;
//                     }
//                     else
//                     {
//                         result[2][0]++;
//                     }
//                 }
//             }
//             // 计算每个类别的F1分数
//             double f1_0 = F1Score(result[0][1], result[0][0], result[1][0] + result[2][0]);
//             double f1_1 = F1Score(result[1][1], result[1][0], result[0][0] + result[2][0]);
//             double f1_2 = F1Score(result[2][1], result[2][0], result[0][0] + result[1][0]);
//             // 计算总的F1分数
//             double total_f1 = f1_0 + f1_1 + f1_2;
//             // 更新最大F1分数和最优阈值
//             if(total_f1 > best_f1)
//             {
//                 best_f1 = total_f1;
//                 best_t1 = therhold1;
//                 best_t2 = therhold2;
//             }
//             fout << "Therhold1: " << therhold1 << "Therholde2: " << therhold2 <<endl;
//             fout << f1_0 << endl << f1_1 << endl << f1_2 << endl;
//             fout<<endl;
//         }
//     }
//     cout << "Best F1: " << best_f1 << " Therhold1: " << best_t1 << " Therhold2: " << best_t2 << endl;
//     return 0;
// }
