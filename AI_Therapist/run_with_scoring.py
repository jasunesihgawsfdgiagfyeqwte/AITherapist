import os
import sys

# 确保能找到backend模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试基本功能...")
    
    try:
        # 导入测试
        from backend.main import BasicAISystem
        print("✅ 模块导入成功")
        
        # 初始化测试
        ai_system = BasicAISystem()
        print("✅ 系统初始化成功")
        
        # 简单对话测试
        result = ai_system.simple_chat("test_user", "我今天心情不好")
        print("✅ 对话功能测试成功")
        
        # 显示结果
        print(f"\n📊 测试结果:")
        print(f"   用户输入: {result['user_input']}")
        print(f"   AI回复: {result['ai_response']}")
        
        if 'lsm_single' in result:
            print(f"   LSM评分: {result['lsm_single']:.4f}")
            print(f"   nCLiD评分: {result['nclid_single']:.4f}")
        
        if 'error' in result:
            print(f"   ⚠️ 评分错误: {result['error']}")
        
        print(f"\n🎉 基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"💡 请检查以下问题:")
        print(f"   1. 所有__init__.py文件是否存在")
        print(f"   2. Python路径是否正确")
        print(f"   3. 依赖包是否安装完整")
        return False

def interactive_mode():
    """交互模式"""
    print("\n🎮 进入交互模式...")
    
    try:
        from backend.main import BasicAISystem
        ai_system = BasicAISystem()
        
        user_id = input("请输入用户名: ").strip() or "test_user"
        print(f"欢迎, {user_id}!")
        print("输入 'exit' 退出程序\n")
        
        while True:
            user_input = input(f"[{user_id}] 你: ").strip()
            
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 再见!")
                break
            
            if not user_input:
                continue
            
            try:
                result = ai_system.simple_chat(user_id, user_input)
                
                print(f"🤖 AI: {result['ai_response']}")
                
                if 'lsm_single' in result:
                    print(f"📊 LSM: {result['lsm_single']:.4f} | "
                          f"nCLiD: {result['nclid_single']:.4f} | "
                          f"轮次: {result['total_rounds']}")
                print()
                
            except Exception as e:
                print(f"❌ 错误: {e}\n")
                
    except Exception as e:
        print(f"❌ 无法启动交互模式: {e}")

if __name__ == "__main__":
    print("🚀 AI Therapist - 基础模块测试")
    print("=" * 50)
    
    # 先测试基本功能
    if test_basic_functionality():
        # 如果基本功能正常，进入交互模式
        interactive_mode()
    else:
        print("\n❌ 基本功能测试失败，请先解决问题")
