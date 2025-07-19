#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Ising.hpp"

namespace py = pybind11;

PYBIND11_MODULE(Ising, m) {
    py::class_<Ising>(m, "Ising")
        // 构造函数
        .def(py::init<int, float>(), py::arg("L"), py::arg("t"))
        
        // 获取温度
        .def("get_temperature", &Ising::get_temperature)
        
        // 运行模拟
        .def("run", &Ising::run, py::arg("Ntest"), py::arg("spacing"))
        
        // 获取能量数据
        .def("get_e", [](const Ising &model) {
            auto vec = model.get_e_data();
            return py::array_t<int>(vec.size(), vec.data());
        })
        
        // 获取磁化数据
        .def("get_m", [](const Ising &model) {
            auto vec = model.get_m_data();
            return py::array_t<int>(vec.size(), vec.data());
        })
        
        // 获取自旋配置（二维数组）
        .def("get_spin", [](const Ising &model) {
            auto spin_flat = model.get_spin();
            int L = static_cast<int>(std::sqrt(spin_flat.size()));
            auto arr = py::array_t<int>({L, L});
            auto buf = arr.mutable_unchecked<2>();
            
            for (int x = 0; x < L; ++x) {
                for (int y = 0; y < L; ++y) {
                    buf(x, y) = spin_flat[x * L + y];
                }
            }
            return arr;
        })
        
        // 设置自旋配置
        .def("set_spin", [](Ising &model, py::array_t<int> arr) {
            auto buf = arr.unchecked<2>();

            int L = buf.shape(0);
            std::vector<int> new_spin(L * L);
            
            for (int x = 0; x < L; ++x) {
                for (int y = 0; y < L; ++y) {
                    new_spin[x * L + y] = buf(x, y) ;
                }
            }
            model.set_spin(new_spin);
        })
        
        // 设置模拟数据
        .def("set_data", &Ising::set_data, py::arg("m_data"), py::arg("e_data"))
        
        // 序列化支持
        .def(py::pickle(
            [](const Ising &model) {  // __getstate__
                return py::make_tuple(
                    model.get_spin(),  // 自旋配置
                    model.get_m_data(), // 磁化数据
                    model.get_e_data(), // 能量数据
                    model.get_temperature()  // 温度
                );
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");
                
                // 从元组中提取数据
                auto spin = t[0].cast<std::vector<int>>();
                auto m_data = t[1].cast<std::vector<int>>();
                auto e_data = t[2].cast<std::vector<int>>();
                float temperature = t[3].cast<float>();
                
                // 计算格点尺寸
                int L = static_cast<int>(std::sqrt(spin.size()));
                
                // 创建新模型
                Ising model(L, temperature);
                model.set_spin(spin);
                model.set_data(m_data, e_data);
                
                return model;
            }
        ));
}