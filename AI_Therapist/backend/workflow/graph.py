from backend.workflow.state import ChatStateWithScoring

class DummyWorkflow:
    def invoke(self, state: ChatStateWithScoring) -> ChatStateWithScoring:
        # 模拟AI回复（真实项目中这里是LangGraph工作流）
        state["ai_response"] = f"收到：{state['user_input']}"
        return state

def create_workflow():
    return DummyWorkflow()
