#!/usr/bin/env python3
"""
Photoye 启动脚本
检查依赖并启动主程序
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要 Python 3.8 或更高版本")
        print(f"   当前版本: Python {sys.version}")
        return False
    else:
        print(f"✅ Python版本检查通过: {sys.version}")
        return True

def check_dependencies():
    """检查必要的依赖包 (V3.0 纯 ONNX 运行时)"""
    dependencies = [
        ('PyQt6', 'PyQt6'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('onnxruntime', 'onnxruntime'),
        ('cv2', 'opencv-contrib-python'),
        ('tokenizers', 'tokenizers'),
        ('pytest', 'pytest'),
    ]
    
    missing_deps = []
    
    print("\n检查依赖包...")
    for import_name, package_name in dependencies:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"❌ 缺少依赖: {package_name}")
            missing_deps.append(package_name)
        else:
            print(f"✅ {package_name} 已安装")
    
    return missing_deps

def install_dependencies(missing_deps):
    """尝试安装缺失的依赖包"""
    if not missing_deps:
        return True
    
    print(f"\n发现 {len(missing_deps)} 个缺失的依赖包")
    user_input = input("是否自动安装？(y/n): ").lower().strip()
    
    if user_input != 'y':
        print("\n请手动安装依赖包:")
        print("pip install -r requirements.txt")
        return False
    
    try:
        print("\n正在安装依赖包...")
        for package in missing_deps:
            print(f"安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装完成")
        
        print("\n所有依赖包安装完成！")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        print("\n请尝试手动安装:")
        print("pip install -r requirements.txt")
        return False

def check_main_files():
    """检查主程序文件是否存在"""
    required_files = ['main.py', 'database.py', 'analyzer.py', 'worker.py']
    missing_files = []
    
    print("\n检查程序文件...")
    for filename in required_files:
        if not Path(filename).exists():
            print(f"❌ 缺少文件: {filename}")
            missing_files.append(filename)
        else:
            print(f"✅ {filename} 存在")
    
    return len(missing_files) == 0

def main():
    """主函数"""
    print("=" * 50)
    print("🎯 Photoye 本地智能照片管理工具")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        input("\n按任意键退出...")
        return
    
    # 检查程序文件
    if not check_main_files():
        print("\n❌ 程序文件不完整，请检查项目结构")
        input("\n按任意键退出...")
        return
    
    # 检查依赖
    missing_deps = check_dependencies()
    
    # 安装缺失的依赖
    if missing_deps:
        if not install_dependencies(missing_deps):
            input("\n按任意键退出...")
            return
    
    print("\n" + "=" * 50)
    print("🚀 启动 Photoye...")
    print("=" * 50)
    
    try:
        # 导入并运行主程序
        from main import main as photoye_main
        photoye_main()
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("\n请确保所有依赖都已正确安装")
        input("\n按任意键退出...")
        
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        input("\n按任意键退出...")

if __name__ == "__main__":
    main()