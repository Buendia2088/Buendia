#include "Complex.cpp"
#include <Eigen/Core>
using namespace std;
using namespace Eigen;

using Cpx = Complex<double>; // Self design complex class
using mtrx = Matrix<Cpx, Dynamic, Dynamic>; // Matrix class that will use in this project

struct BaseComponent // Virtual Base Class for Components
{
    string Name;
    size_t BranchNo;
    size_t PositivePin;
    size_t NegativePin;
    Cpx State; // Value of the component: Ohm for Impedance, A for CurrentSource, V for Voltage Source

    enum class Type : unsigned
    {
        Impedance = 0, // Z = a + bj
        CurrentSource = 1, // I_s = a + bj
        VoltageSource = 2, // U_s = a + bj
        CCCS = 3, // I_s = a * I_x
        VCCS = 4, // U_s = a * U_x
        CCVS = 5, // I_s = a * U_x
        VCVS = 6, // U_s = a * I_x
    };

    virtual ~BaseComponent() = default;

    virtual unsigned int GetType() = 0;

    virtual void debug() = 0;

    virtual bool isImpedance() { return false; }

    virtual bool isCurrentSource() { return false; }

    virtual bool isVoltageSource() { return false; }

    virtual bool isCCCS() { return false; }

    virtual bool isVCCS() { return false; } 

    virtual bool isCCVS() { return false; } 

    virtual bool isVCVS() { return false; }

    virtual bool isControlSource() { return false; }

    virtual void action(mtrx &Cof, mtrx &Rhs) = 0;
};

struct Circuit;

optional<BaseComponent*> FindByBranchNo(const Circuit& Cmp ,size_t BranchNo);

struct Impendence : BaseComponent // Component Except Control Source
{
    const Circuit& CR; // Circuit that the impendence belongs to

    Impendence(string name, size_t cn ,size_t pos, size_t neg, Cpx state,const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = state;
    }

    ~Impendence() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::Impedance);
    }

    bool isImpedance() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"State: "<<State<<endl;
        cout<<"Type: "<<"Impedance"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        // Self Impendence
        Cof(PositivePin, PositivePin) += 1.0 / State;
        Cof(NegativePin, NegativePin) += 1.0 / State;
        
        // Mutual Impendence
        Cof(NegativePin, PositivePin) -= 1.0 / State;
        Cof(PositivePin, NegativePin) -= 1.0 / State;
    }
};

struct CurrentSource : BaseComponent // Component Except Control Source
{
    const Circuit& CR; // Circuit that the current source belongs to

    CurrentSource(string name, size_t cn, size_t pos, size_t neg, Cpx state,const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = state;
    }

    ~CurrentSource() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::CurrentSource);
    }

    bool isCurrentSource() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"State: "<<State<<endl;
        cout<<"Type: "<<"CurrentSource"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        Rhs(PositivePin) += State;
        Rhs(NegativePin) -= State;
    }
};

struct VoltageSource : BaseComponent // Component Except Control Source
{
    size_t AVNo = 0; // The number of the new independent variable
    const Circuit& CR; // Circuit that the voltage source belongs to

    VoltageSource(string name, size_t cn, size_t pos, size_t neg, Cpx state, const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = state;
    }

    ~VoltageSource() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::VoltageSource);
    }

    bool isVoltageSource() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"State: "<<State<<endl;
        cout<<"Type: "<<"VoltageSource"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        // Pretend that the voltage source is a current source
        Cof(PositivePin, AVNo) -= scalar(1.0);
        Cof(NegativePin, AVNo) += scalar(1.0);

        // Add VCR
        Cof(AVNo, PositivePin) += scalar(1.0);
        Cof(AVNo, NegativePin) -= scalar(1.0);
        Rhs(AVNo) = State;
    }
};

struct CCCS : BaseComponent // Current Controlled Current Source
{
    size_t ControlledBranchNo; // Branch Number of the controlled current source
    const Circuit& CR; // Circuit that the CCCS belongs to

    CCCS(string name, size_t cn, size_t pos, size_t neg, Cpx gain, size_t cbn, const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = gain;
        ControlledBranchNo = cbn;
    }

    ~CCCS() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::CCCS);
    }

    bool isCCCS() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"Gain: "<<State<<endl;
        cout<<"Controlled Branch No: "<<ControlledBranchNo<<endl;
        cout<<"Type: "<<"CCCS"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        auto Cmp_O = FindByBranchNo(CR ,ControlledBranchNo);

        if(Cmp_O== nullopt) throw runtime_error("Controlled Branch Not Found");

        auto* Cmp = Cmp_O.value();

        if(Cmp->isImpedance())
        {
            auto ControlledImpledance = Cmp->State;
            if(ControlledImpledance == Cpx(0)) throw runtime_error("Devived by Zero: Controlled Source is short circuit");

            Cof(PositivePin, Cmp->PositivePin) -= State / ControlledImpledance;
            Cof(PositivePin, Cmp->NegativePin) += State / ControlledImpledance;
            Cof(NegativePin, Cmp->PositivePin) += State / ControlledImpledance;
            Cof(NegativePin, Cmp->NegativePin) -= State / ControlledImpledance;
        }
        else if(Cmp->isCurrentSource())
        {
            // ! Might have serious problem here
            Rhs(PositivePin) += State * Cmp->State;
            Rhs(NegativePin) -= State * Cmp->State;
        }
        else if(Cmp->isVoltageSource())
        {
            // Double Check here
            auto* Target = dynamic_cast<VoltageSource*>(Cmp);
            Cof(PositivePin, Target->AVNo) -= State;
            Cof(NegativePin, Target->AVNo) += State;
        }
        else throw runtime_error("Controlled Target Type Not Supported"); // If the controlled target is another control source, throw an error
    }
};

struct VCCS : BaseComponent // Voltage Controlled Current Source
{
    size_t PositiveControlPin; // Positive Control Pin
    size_t NegativeControlPin; // Negative Control Pin
    const Circuit& CR; // Circuit that the VCCS belongs to

    VCCS(string name, size_t cn, size_t pos, size_t neg, Cpx gain, size_t pcp, size_t ncp, const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = gain;
        PositiveControlPin = pcp;
        NegativeControlPin = ncp;
    }

    ~VCCS() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::VCCS);
    }

    bool isVCCS() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"Gain: "<<State<<endl;
        cout<<"Positive Control Pin: "<<PositiveControlPin<<endl;
        cout<<"Negative Control Pin: "<<NegativeControlPin<<endl;
        cout<<"Type: "<<"VCCS"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        // Pretend that the VCCS is a current source
        Cof(PositivePin, PositiveControlPin) -= State;
        Cof(PositivePin, NegativeControlPin) += State;
        Cof(NegativePin, NegativeControlPin) -= State;
        Cof(NegativePin, PositiveControlPin) += State;
    }
};  
struct CCVS : BaseComponent // Current Controlled Voltage Source
{
    size_t ControlledBranchNo; // Branch Number of the controlled current source
    size_t AVNo; // The number of the new independent variable
    const Circuit& CR; // Circuit that the CCVS belongs to

    CCVS(string name, size_t cn, size_t pos, size_t neg, Cpx gain, size_t cbn, const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = gain;
        ControlledBranchNo = cbn;
    }

    ~CCVS() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::CCVS);
    }

    bool isCCVS() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"Gain: "<<State<<endl;
        cout<<"Controlled Branch No: "<<ControlledBranchNo<<endl;
        cout<<"Type: "<<"CCVS"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        auto Cmp_O = FindByBranchNo(CR, ControlledBranchNo);

        if(Cmp_O == nullptr) throw runtime_error("Controlled Branch Not Found");

        auto* Cmp = Cmp_O.value();

        if(Cmp->isImpedance())
        {
            auto ControlledImpledance = Cmp->State;
            if(ControlledImpledance == Cpx(0)) throw runtime_error("Devived by Zero: Controlled Source is short circuit");

            Cof(PositivePin, AVNo) -= scalar(1.0);
            Cof(NegativePin, AVNo) += scalar(1.0);
            Cof(AVNo, Cmp->PositivePin) -= State / ControlledImpledance;
            Cof(AVNo, Cmp->NegativePin) += State / ControlledImpledance;
            Cof(AVNo, PositivePin) += scalar(1.0);
            Cof(AVNo, NegativePin) -= scalar(1.0);

        }
        else if(Cmp->isCurrentSource())
        {
            // Really?
            Cof(PositivePin, AVNo) -= scalar(1.0);
            Cof(NegativePin, AVNo) += scalar(1.0);

            Cof(AVNo, PositivePin) += scalar(1.0);
            Cof(AVNo, NegativePin) -= scalar(1.0);
            Rhs(AVNo) = State * Cmp->State;
        }
        else if(Cmp->isVoltageSource())
        {
            // Double Check here
            auto* Target = dynamic_cast<VoltageSource*>(Cmp);
            Cof(PositivePin, AVNo) -= scalar(1.0);
            Cof(NegativePin, AVNo) += scalar(1.0);
            
            Cof(AVNo, PositivePin) += scalar(1.0);
            Cof(AVNo, NegativePin) -= scalar(1.0);
            Cof(AVNo, Target->AVNo) -= State;
        }
        else throw runtime_error("Controlled Target Type Not Supported"); // If the controlled target is another control source, throw an error
    }
};

struct VCVS : BaseComponent // Voltage Controlled Voltage Source
{
    size_t PositiveControlPin; // Positive Control Pin
    size_t NegativeControlPin; // Negative Control Pin
    size_t AVNo; // The number of the new independent variable
    const Circuit& CR; // Circuit that the VCVS belongs to

    VCVS(string name, size_t cn, size_t pos, size_t neg, Cpx gain, size_t pcp, size_t ncp, const Circuit& cr) : CR(cr)
    {
        Name = name;
        BranchNo = cn;
        PositivePin = pos;
        NegativePin = neg;
        State = gain;
        PositiveControlPin = pcp;
        NegativeControlPin = ncp;
    }

    ~VCVS() = default;

    unsigned int GetType()
    {
        return static_cast<unsigned int>(Type::VCVS);
    }

    bool isVCVS() { return true; }

    void debug()
    {
        cout<<"Name: "<<Name<<endl;
        cout<<"Branch No: "<<BranchNo<<endl;
        cout<<"Positive Pin: "<<PositivePin<<endl;
        cout<<"Negative Pin: "<<NegativePin<<endl;
        cout<<"Gain: "<<State<<endl;
        cout<<"Positive Control Pin: "<<PositiveControlPin<<endl;
        cout<<"Negative Control Pin: "<<NegativeControlPin<<endl;
        cout<<"Type: "<<"VCVS"<<endl;
    }

    void action(mtrx &Cof, mtrx &Rhs)
    {
        // Pretend that the VCVS is a voltage source
        Cof(PositivePin, AVNo) -= scalar(1.0);
        Cof(NegativePin, AVNo) += scalar(1.0);

        // Add VCR
        Cof(AVNo, PositivePin) += scalar(1.0);
        Cof(AVNo, NegativePin) -= scalar(1.0);
        Cof(AVNo, PositiveControlPin) -= State;
        Cof(AVNo, NegativeControlPin) += State;
    }
};