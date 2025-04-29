from hstest import dynamic_test, StageTest, CheckResult, TestedProgram
import re


class Test(StageTest):

    responses = []
    prompts = [
        {"prompt": "What is the largest ocean?", "answer": "Pacific Ocean"},
        {"prompt": "What is 15 + 25?", "answer": "40"},
        {"prompt": "What is the capital of France?", "answer": "Paris"},
        {"prompt": "Which color is this: #000?", "answer": "black"},
        {"prompt": "End conversation", "answer": ""},
    ]
    call_id_regex = r"call_[A-Za-z0-9]{24}|chatcmpl-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

    @dynamic_test(time_limit=60000)
    def test1(self):
        program = TestedProgram()
        output = program.start().strip()

        for run in self.prompts:
            prompt = run["prompt"]
            answer = run["answer"]

            # Check if output is "Enter a message: "
            if "Enter a message:" not in output:
                return CheckResult.wrong("The program should output 'Enter a message: ' each time.")

            if not program.is_waiting_input():
                return CheckResult.wrong("The program should be waiting for input.")

            output = program.execute(prompt).strip()

            # Check if output has the prompt with You:
            if 'You: ' + prompt not in output:
                return CheckResult.wrong("The prompt was not found in the output. You should output the prompt with 'You: ' before it.\nYour output:" + output )

            # Check if output has the assistant response with Assistant:
            if 'Assistant:' not in output:
                return CheckResult.wrong("The assistant response was not found in the output. You should output the assistant response with 'Assistant: ' before it.\nYour output:" + output)

            # Check if the response is correct
            if answer.lower() not in output.lower():
                return CheckResult.wrong("The assistant's response doesn't contain the expected answer for the prompt.\nYour output:" + output)

            # Check if output has the cost
            if not re.search(r'Cost: \$\d+\.\d+', output):
                return CheckResult.wrong("The cost was not found in the output. You should output the cost with 'Cost: $' before it.\nYour output:" + output)

            # Check if response is different from the previous one
            if output in self.responses:
                return CheckResult.wrong("The response is the same as the previous one. "
                                         "It should be different each time.")
            self.responses.append(output)

        # Check if program called the function to finish
        if not re.search(self.call_id_regex, output):
            return CheckResult.wrong("The program should call the function to finish the conversation and output should have the call id.")

        if program.is_finished():
            return CheckResult.correct()
        else:
            return CheckResult.wrong("The program should end after it's asked to finish.")


if __name__ == '__main__':
    Test('main').run_tests()