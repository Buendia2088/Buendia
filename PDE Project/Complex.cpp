#include <iostream>
#include <cmath>
#include <regex>
using namespace std;

#ifndef COMPLEX_H
#define COMPLEX_H

template <typename T>
T epsilon()
{
    return T(1);
}

template <>
double epsilon<double>()
{
    return 1e-12;
}

// Self design complex class
template <typename T>
class Complex {
        T REAL, IMAG; // Real part and imaginary part
    public:
        
        typedef T Real;
        Complex() : REAL(0), IMAG(0) {}
        Complex(T r, T i) : REAL(r), IMAG(i) {}
        Complex(const Complex& other) : REAL(other.REAL), IMAG(other.IMAG) {}
        Complex(Complex&& other) : REAL(other.REAL), IMAG(other.IMAG) {}
        Complex(const T& other) : REAL(other), IMAG(0) {}
        Complex(const complex<double>& other) : REAL(other.real()), IMAG(other.imag()) {}
        Complex(const string& str) // Accept string like "1+j2"
        {
            regex re("([+-]?[0-9]*\\.?[0-9]+)?([+-]?j[0-9]*\\.?[0-9]*)?");
            smatch match;
            if (regex_match(str, match, re)) {
                if (match[1].matched) 
                {
                    try
                    {
                        REAL = stod(match[1]);
                    }
                    catch(invalid_argument& e)
                    {
                        cerr<<"Invalid Real Part: "<<match[1]<<endl;
                        REAL = 0;
                    }
                    REAL = stod(match[1]);
                } 
                else 
                {
                    REAL = 0;
                }

                if (match[2].matched) 
                {
                    string imag_str = match[2];
                    imag_str.erase(imag_str.find('j'), 1); // Remove the 'j'
                    if (imag_str.empty() || imag_str == "+" || imag_str == "-") 
                    {
                        IMAG = (imag_str == "-") ? -1 : 1;
                    } 
                    else 
                    {
                        try
                        {
                            IMAG = stod(imag_str);
                        }
                        catch(invalid_argument& e)
                        {
                            cerr<<"Invalid Imaginary Part: "<<imag_str<<endl;
                            IMAG = 0;
                        }
                    }
                } 
                else 
                {
                    IMAG = 0;
                }
            } else {
                throw runtime_error("Invalid Complex Number Format");
            }
        }

        T real() const { return REAL; }
        T imag() const { return IMAG; }

        Complex conjugate() const {
            return Complex(REAL, -IMAG);
        }

        // Normal operators
        Complex operator+(const Complex& other) const {
            return Complex(REAL + other.REAL, IMAG + other.IMAG);
        }

        Complex operator-(const Complex& other) const {
            return Complex(REAL - other.REAL, IMAG - other.IMAG);
        }

        Complex operator*(const Complex& other) const {
            return Complex(REAL * other.REAL - IMAG * other.IMAG,
                        REAL * other.IMAG + IMAG * other.REAL);
        }

        Complex operator/(const Complex& other) const {
            T denominator = other.REAL * other.REAL + other.IMAG * other.IMAG;
            return Complex((REAL * other.REAL + IMAG * other.IMAG) / denominator,
                        (IMAG * other.REAL - REAL * other.IMAG) / denominator);
        }
        
        // Assignment operators
        Complex& operator=(const Complex& other) {
            REAL = other.REAL;
            IMAG = other.IMAG;
            return *this;
        }

        Complex& operator=(Complex&& other) {
            REAL = other.REAL;
            IMAG = other.IMAG;
            return *this;
        }

        // Compound assignment operators
        Complex& operator+=(const Complex& other) {
            REAL += other.REAL;
            IMAG += other.IMAG;
            return *this;
        }

        Complex& operator-=(const Complex& other) {
            REAL -= other.REAL;
            IMAG -= other.IMAG;
            return *this;
        }

        Complex& operator*=(const Complex& other) {
            T temp = REAL;
            REAL = REAL * other.REAL - IMAG * other.IMAG;
            IMAG = temp * other.IMAG + IMAG * other.REAL;
            return *this;
        }

        Complex& operator/=(const Complex& other) {
            T denominator = other.REAL * other.REAL + other.IMAG * other.IMAG;
            T temp = REAL;
            REAL = (REAL * other.REAL + IMAG * other.IMAG) / denominator;
            IMAG = (IMAG * other.REAL - temp * other.IMAG) / denominator;
            return *this;
        }

        // Real op Complex
        friend Complex operator+(const T& lhs, const Complex& rhs) {
            return Complex(lhs + rhs.REAL, rhs.IMAG);
        }

        friend Complex operator-(const T& lhs, const Complex& rhs) {
            return Complex(lhs - rhs.REAL, -rhs.IMAG);
        }

        friend Complex operator*(const T& lhs, const Complex& rhs) {
            return Complex(lhs * rhs.REAL, lhs * rhs.IMAG);
        }

        friend Complex operator/(const T& lhs, const Complex& rhs) {
            T denominator = rhs.REAL * rhs.REAL + rhs.IMAG * rhs.IMAG;
            return Complex(lhs * rhs.REAL / denominator, -lhs * rhs.IMAG / denominator);
        }

        bool operator==(const Complex& other) const {
            return REAL == other.REAL && IMAG == other.IMAG;
        }

        bool operator!=(const Complex& other) const {
            return REAL != other.REAL || IMAG != other.IMAG;
        }
        
        void printAsPolar() const {
            T r = sqrt(REAL * REAL + IMAG * IMAG);
            T theta = atan(IMAG / REAL);
            cout << r << " <" << theta << "deg" << endl;
        }

        friend ostream& operator<<(ostream& os, const Complex& c) {
            // os << c.REAL << (c.IMAG >= 0 ? "+" : "-") <<"j"<<(c.IMAG >= 0 ? c.IMAG : -c.IMAG);
            if(fabs(c.REAL) > epsilon<T>())
            {
                os<<c.REAL;
            }
            else
            {
                os<<"0";
            }
            if(fabs(c.IMAG) > epsilon<T>())
            {
                if(c.IMAG > 0)
                {
                    os<<"+";
                }
                else
                {
                    os<<"-";
                }
                os<<"j"<<fabs(c.IMAG);
            }
            else
            {
                os<<"+0j";
            }
            return os;
        }

        friend istream& operator>>(istream& is, Complex& c) {
            string str;
            is >> str;
            c = Complex(str);
            return is;
        }

        friend Complex log10(const Complex& c) {
            T r = sqrt(c.REAL * c.REAL + c.IMAG * c.IMAG);
            T theta = atan2(c.IMAG, c.REAL);
            return Complex(std::log10(r), theta / std::log(10));
        }

        friend Complex log(const Complex& c) {
            T r = sqrt(c.REAL * c.REAL + c.IMAG * c.IMAG);
            T theta = atan2(c.IMAG, c.REAL);
            return Complex(std::log(r), theta);
        }

        friend Complex exp(const Complex& c) {
            T r = std::exp(c.REAL);
            return Complex(r * std::cos(c.IMAG), r * std::sin(c.IMAG));
        }

        friend Complex sqrt(const Complex& c) {
            T r = sqrt(c.REAL * c.REAL + c.IMAG * c.IMAG);
            T theta = atan2(c.IMAG, c.REAL);
            return Complex(sqrt(r) * cos(theta / 2), sqrt(r) * sin(theta / 2));
        }

        operator complex<double>() const {
            return complex<double>(REAL, IMAG);
        }
};

template <typename T>
Complex<T> scalar(T r) {
    return Complex<T>(r, 0);
}

template <typename T>
T real (const Complex<T>& c) {
    return c.real();
}

template <typename T>
T imag (const Complex<T>& c) {
    return c.imag();
}

#include <Eigen/Core>
namespace Eigen {
template<>
struct NumTraits<Complex<double>> : GenericNumTraits<Complex<double>> {
    typedef Complex<double> Real;
    typedef Complex<double> NonInteger;
    typedef Complex<double> Nested;

    static inline Real epsilon() { return std::numeric_limits<double>::epsilon(); }
    static inline Real dummy_precision() { return 1e-12; }
    static inline int digits10() { return std::numeric_limits<double>::digits10; }

    enum {
        IsComplex = 1,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};
} // namespace Eigen

#ifndef Main
    int main()
    {
        Complex<double> C;
        cin>>C;
        cout << 1.0 / C << endl;
    }
#endif

#endif