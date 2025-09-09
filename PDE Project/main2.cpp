#define Main
#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Dense>
#include<nlohmann/json.hpp>
#include<optional>
#include "Complex.cpp"
#include "Components.cpp"
using namespace std;
using namespace Eigen;

using Cpx = Complex<double>; // Self design complex class
using mtrx = Matrix<Cpx, Dynamic, Dynamic>; // Matrix class that will use in this project

struct Circuit // Circuit Class
{
    vector<BaseComponent*> Components;
    vector<unsigned int> Count;
    double Frequency; // omega, not mu;

    Circuit() : Components(), Count(8, 0) {}

    ~Circuit()
    {
        for(auto&& iter : Components)
        {
            delete iter;
        }
    }

    void AddComponent(BaseComponent *Cmp)
    {
        Components.push_back(Cmp);

        Count[Cmp->GetType()]++;
    }

    /*
        Sort the components by order as:
        Impedance = 0,
        CurrentSource = 1,
        VoltageSource = 2,
        CCCS = 3,
        VCCS = 4,
        CCVS = 5,
        VCVS = 6,
        while the same type of components are sorted by BranchNo
        so that it will be easier to construct the equation
    */
    void Sort()
    {
        sort(Components.begin(), Components.end(), 
        [](BaseComponent *a, BaseComponent *b) 
        { 
            return a->GetType() < b->GetType() or (a->GetType() == b->GetType() and a->BranchNo < b->BranchNo);
        });
    }

    optional<BaseComponent*> FindByName(string name) const
    {
        for(auto&& iter : Components)
        {
            if(iter->Name == name) return iter;
        }
        return nullopt;
    }

    [[deprecated("Use FindByBranchNo instead")]] optional<BaseComponent*> FindByPin(size_t pos, size_t neg) const 
    {
        for(auto&& iter : Components)
        {
            if(iter->PositivePin == pos and iter->NegativePin == neg) return iter;
        }
        return nullopt;
    } // Multiple components may have the same pins

    optional<BaseComponent*> FindByBranchNo(size_t BranchNo) const
    {
        for(auto&& iter : Components)
        {
            if(iter->BranchNo == BranchNo) return iter;
        }
        return nullopt;
    }

    void debug()
    {
        for(auto&& iter : Components)
        {
            iter->debug();
        }
        cout<<"Frequency: "<<Frequency<<endl;
        cout<<"Component Count: "<<Components.size()<<endl;
        cout<<"Impedance: "<<Count[0]<<endl;
        cout<<"Current Source: "<<Count[1]<<endl;
        cout<<"Voltage Source: "<<Count[2]<<endl;
        cout<<"CCCS: "<<Count[3]<<endl;
        cout<<"VCCS: "<<Count[4]<<endl;
        cout<<"CCVS: "<<Count[5]<<endl;
        cout<<"VCVS: "<<Count[6]<<endl;
    }

    friend istream& operator>>(istream& is, Circuit& self);
};

optional<BaseComponent*> FindByBranchNo(const Circuit& Cmp, size_t BranchNo)
{
    return Cmp.FindByBranchNo(BranchNo);
}

void ToEq(const Circuit& Cmp, mtrx &Cof, mtrx &Rhs)
{
    // Cof.setZero();
    // Rhs.setZero();

    for(auto&& iter : Cmp.Components)
    {
        iter->action(Cof, Rhs);
    }
}

void PreProcess(Circuit& Cmp)
{
    Cmp.Sort();
    size_t AVNo = Cmp.Components.size();
    for(auto&& iter : Cmp.Components)
    {
        if(iter->isVoltageSource())
        {
            dynamic_cast<VoltageSource*>(iter)->AVNo = AVNo++;
        }
        else if(iter->isVCVS())
        {
            dynamic_cast<VCVS*>(iter)->AVNo = AVNo++;
        }
        else if(iter->isCCVS())
        {
            dynamic_cast<CCVS*>(iter)->AVNo = AVNo++;
        } 
    }
}

void parseFromJson(Circuit& Cmp, const string& filename = "input.json");

void printToJson(const Circuit& Cmp, const VectorXcd& res, const string& filename = "output.json")
{
    using nlohmann::json;
    json j;
    j["n"] = Cmp.Components.size();
    j["frequency"] = Cmp.Frequency;
    j["node_voltage"] = json::array();
    j["component_result"] = json::array();
    vector<unsigned short> used_node(Cmp.Components.size() , 0);
    for(auto&& iter : Cmp.Components)
    {
        // Assert(iter->PositivePin < n);
        // Assert(iter->NegativePin < n);
        used_node[iter->PositivePin] = 1;
        used_node[iter->NegativePin] = 1;
    }

    for(size_t i = 0; i < Cmp.Components.size(); i++)
    {
        if(used_node[i] == 1)
        j["node_voltage"].push_back({i, Cpx(res(i, 0)).real(), Cpx(res(i, 0)).imag()});
    }
    
    for(auto&& iter : Cmp.Components)
    {
        auto Vol = res(iter->PositivePin, 0) - res(iter->NegativePin, 0);
        Cpx Cur{};
        if(iter->isVoltageSource())
        {
            Cur = res[dynamic_cast<VoltageSource*>(iter)->AVNo];
        }
        else if(iter->isVCVS())
        {
            Cur = res[dynamic_cast<VCVS*>(iter)->AVNo];
        }
        else if(iter->isCCVS())
        {
            Cur = res[dynamic_cast<CCVS*>(iter)->AVNo];
        }
        else if(iter->isCurrentSource())
        {
            Cur = iter->State;
        }
        else if(iter->isImpedance())
        {
            Cur = Cpx(Vol) / iter->State;
        }
        else if(iter->isVCCS())
        {
            Cur = iter->State * (res[dynamic_cast<VCCS*>(iter)->PositiveControlPin] - res[dynamic_cast<VCCS*>(iter)->NegativeControlPin]);
        }
        else if(iter->isCCCS())
        {
            auto Cmp_O = FindByBranchNo(Cmp, dynamic_cast<CCCS*>(iter)->ControlledBranchNo);
            if(Cmp_O == nullopt) throw runtime_error("Controlled Branch Not Found");
            auto* ColCmp = Cmp_O.value();
            if(ColCmp->isImpedance())
            {
                Cur = iter->State * (res[ColCmp->PositivePin] - res[ColCmp->NegativePin]) / ColCmp->State;
            }
            else if(ColCmp->isCurrentSource())
            {
                Cur = iter->State * ColCmp->State;
            }
            else if(ColCmp->isVoltageSource())
            {
                Cur = iter->State * (res[dynamic_cast<VoltageSource*>(ColCmp)->AVNo]);
            }
            else throw runtime_error("Controlled Target Type Not Supported");
        }
        else throw runtime_error("Component Type Not Supported");

        j["component_result"].push_back({iter->Name, Vol.real() , Vol.imag() , Cur.real(), Cur.imag()});
    }

    ofstream fout(filename);
    if(!fout.is_open())
    {
        throw runtime_error("I/O error: Can not open " + filename);
    }
    fout<<j.dump(4);
    fout.close();
}

void printToStd(const Circuit& Cmp, const MatrixXcd& res)
{
    vector<unsigned short> used_node(Cmp.Components.size() , 0);
    for(auto&& iter : Cmp.Components)
    {
        // Assert(iter->PositivePin < n);
        // Assert(iter->NegativePin < n);
        used_node[iter->PositivePin] = 1;
        used_node[iter->NegativePin] = 1;
    }
    cout<<"Node Voltage: "<<endl;
    for(size_t i = 1; i < Cmp.Components.size(); i++)
    {
        if(used_node[i] == 1)
        {
            cout<<"Node "<<i<<": "<<Cpx(res(i, 0))<<endl;
        }
    }
    cout<<endl;
    cout<<"Component result: "<<endl;
    for(auto&& iter : Cmp.Components)
    {
        if(iter->isVoltageSource())
        {
            cout<<iter->Name<<": "<<Cpx(res(dynamic_cast<VoltageSource*>(iter)->AVNo, 0))<<" A"<<endl;
        }
        else if(iter->isVCVS())
        {
            cout<<iter->Name<<": "<<Cpx(res(dynamic_cast<VCVS*>(iter)->AVNo, 0))<<" A"<<endl;
        }
        else if(iter->isCCVS())
        {
            cout<<iter->Name<<": "<<Cpx(res(dynamic_cast<CCVS*>(iter)->AVNo, 0))<<" A"<<endl;
        }
        cout<<iter->Name<<": "<<Cpx(res(iter->PositivePin, 0) - res(iter->NegativePin, 0))<<" V"<<endl;
    }
}

void TakeAction()
{
    Circuit Cmp;
    // cin>>Cmp;

    parseFromJson(Cmp);

    Cmp.debug();

    PreProcess(Cmp);
    size_t Rows = Cmp.Components.size() + Cmp.Count[2] + Cmp.Count[6] + Cmp.Count[5];

    mtrx Cof(Rows + 1, Rows);
    mtrx Rhs(Rows + 1, 1);
    Cof(Rows, 0) = scalar(1.0);


    ToEq(Cmp, Cof, Rhs);

    cout<<Cof<<endl;
    cout<<Rhs<<endl;

    MatrixXcd COF = Cof.cast<complex<double>>();
    MatrixXcd RHS = Rhs.cast<complex<double>>();
    cout<<COF<<endl;
    cout<<RHS<<endl;
    VectorXcd res = COF.fullPivLu().solve(RHS);
    printToJson(Cmp, res);
    printToStd(Cmp, res);
}

istream& operator>>(istream& is, Circuit& self)
{
    size_t ComponentCount;
    is>>ComponentCount;
    is>>self.Frequency;
    for(size_t i = 0; i < ComponentCount; i++)
    {
        string Type;
        is>>Type;
        if(Type == "Z" or Type == "Impedance" or Type == "R" or Type == "Resistor")
        {
            string Name;
            size_t Pos, Neg;
            Cpx State;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>State;
            self.AddComponent(new Impendence(Name, BranchNo, Pos, Neg, State, self));
        }
        else if(Type == "C" or Type == "Capacitance")
        {
            string Name;
            size_t Pos, Neg;
            double C;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>C;
            self.AddComponent(new Impendence(Name, BranchNo, Pos, Neg, Cpx(0, - 1.0 / self.Frequency / C), self));
        }
        else if(Type == "L" or Type == "Inductance")
        {
            string Name;
            size_t Pos, Neg;
            double L;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>L;
            self.AddComponent(new Impendence(Name, BranchNo, Pos, Neg, Cpx(0, self.Frequency * L), self));
        }
        else if(Type == "I" or Type == "CS" or Type == "CurrentSource")
        {
            string Name;
            size_t Pos, Neg;
            Cpx State;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>State;
            self.AddComponent(new CurrentSource(Name, BranchNo, Pos, Neg, State, self));
        }
        else if(Type == "U" or Type == "VS" or Type == "VoltageSource")
        {
            string Name;
            size_t Pos, Neg;
            Cpx State;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>State;
            self.AddComponent(new VoltageSource(Name, BranchNo, Pos, Neg, State, self));
        }
        else if(Type == "CCCS")
        {
            string Name;
            size_t Pos, Neg;
            Cpx Gain;
            size_t ControlledBranchNo;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>ControlledBranchNo>>Gain;
            self.AddComponent(new CCCS(Name, BranchNo, Pos, Neg, Gain, ControlledBranchNo, self));
        }
        else if(Type == "VCCS")
        {
            string Name;
            size_t Pos, Neg;
            Cpx Gain;
            size_t PosControl, NegControl;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>PosControl>>NegControl>>Gain;
            self.AddComponent(new VCCS(Name, BranchNo, Pos, Neg, Gain, PosControl, NegControl, self));
        }
        else if(Type == "CCVS")
        {
            string Name;
            size_t Pos, Neg;
            Cpx Gain;
            size_t ControlledBranchNo;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>ControlledBranchNo>>Gain;
            self.AddComponent(new CCVS(Name, BranchNo, Pos, Neg, Gain, ControlledBranchNo, self));
        }
        else if(Type == "VCVS")
        {
            string Name;
            size_t Pos, Neg;
            Cpx Gain;
            size_t PosControl, NegControl;
            size_t BranchNo = i;
            is>>Name;
            #ifdef InputPlanA
                is >> BranchNo;
            #endif
            is>>Pos>>Neg>>PosControl>>NegControl>>Gain;
            self.AddComponent(new VCVS(Name, BranchNo, Pos, Neg, Gain, PosControl, NegControl, self));
        }
        else
        {
            i--;
            cerr<<"Unknown Component Type: "<<Type<<", Try again."<<endl;
        }    
    }
    return is;
}

void parseFromJson(Circuit& Cmp, const string& filename)
{
    using nlohmann::json;
    nlohmann::json j;
    ifstream fin(filename);
    if(!fin.is_open())
    {
        throw runtime_error("I/O Error: Cannot open "+filename);
    }

    try
    {
        fin>>j;
    }
    catch(json::parse_error& e)
    {
        throw runtime_error("Json Parse Error: "+string(e.what()));
    }
    fin.close();
    
    try
    {
        Cmp.Frequency = j["frequency"];
        for(auto&& iter : j["components"])
        {
            string Type = iter["type"];
            if(Type == "R" or Type == "Resistor" or Type == "Z" or Type == "Impedance")
            {
                Cmp.AddComponent(new Impendence(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), Cmp));
            }
            else if(Type == "CS" or Type == "CurrentSource")
            {
                Cmp.AddComponent(new CurrentSource(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), Cmp));
            }
            else if(Type == "VS" or Type == "VoltageSource")
            {
                Cmp.AddComponent(new VoltageSource(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), Cmp));
            }
            else if(Type == "CCCS")
            {
                Cmp.AddComponent(new CCCS(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), iter["col_branch_id"], Cmp));
            }
            else if(Type == "VCCS")
            {
                Cmp.AddComponent(new VCCS(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), iter["pos_col_pin"], iter["neg_col_pin"], Cmp));
            }
            else if(Type == "CCVS")
            {
                Cmp.AddComponent(new CCVS(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), iter["col_branch_id"], Cmp));
            }
            else if(Type == "VCVS")
            {
                Cmp.AddComponent(new VCVS(iter["name"], iter["branch_id"], iter["pos_pin"], iter["neg_pin"], Cpx(iter["workload"].get<string>()), iter["pos_col_pin"], iter["neg_col_pin"], Cmp));
            }
            else
            {
                throw runtime_error("Unknown Component Type: "+Type);
            }
        }
        PreProcess(Cmp);
    }
    catch(json::type_error& e)
    {
        throw runtime_error("Json Type Error: "+string(e.what()));
    }
    catch(json::out_of_range& e)
    {
        throw runtime_error("Json Out of Range: "+string(e.what()));
    }
    catch(json::exception& e)
    {
        throw runtime_error("Json Error: "+string(e.what()));
    }
}

int main()
{
    TakeAction();
    return 0;
}