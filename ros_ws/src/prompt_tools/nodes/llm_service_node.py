#! /usr/bin/env python3

from typing import Any, List, Tuple, Dict
import os, sys
import rospy
from prompt_tools.msg import Prompt
from prompt_tools.srv import ActionsRate, ActionsRateResponse
from prompt_tools.srv import DoneTask, DoneTaskResponse
from std_srvs.srv import Empty, EmptyResponse
import openai

openai.api_key = 'API key'
openai.api_base = "http://127.0.0.1:8080"

class LLMServiceNode:
    prompt_topic_name = "/alpaca/prompt/actions"
    rate_srv_name = "/alpaca/prompt/rate"
    add_done_task_srv_name = "/alpaca/prompt/add_done_task"
    reset_done_tasks_srv_name = "/alpaca/prompt/reset_done_tasks"
    model_name = "llama-7B-4b"

    def __init__(self):
        self._prompt_subscriber = rospy.Subscriber(self.prompt_topic_name, Prompt, queue_size=2, callback=self._prompt_cb)
        self._rate_service = rospy.Service(self.rate_srv_name, ActionsRate, self._rate_service_cb)
        self._add_done_task_service = rospy.Service(self.add_done_task_srv_name, DoneTask, self._add_done_task_cb)
        self._reset_done_tasks_service = rospy.Service(self.reset_done_tasks_srv_name, Empty, self._reset_done_tasks_cb)
        self._prompt_stamp = None
        self._prompt = None
        self._done_tasks = []

    def _prompt_cb(self, msg):
        self._prompt_stamp = msg.header.stamp
        self._prompt = msg

    def _rate_service_cb(self, req):
        stamp = req.stamp
        task = req.task
        rate = rospy.Rate(30)
        # wait for prompt to be published and be more recent than the request. If it equals to 0 send it immediately
        while True:
            if self._prompt_stamp is not None:
                if stamp == rospy.Time(0):
                    rospy.logwarn("requested stamp is zero, so don't waiting for recent prompt")
                    break
                if self._prompt_stamp > stamp:
                    break
            else:
                rospy.logwarn("waiting for prompt to be published")
            rate.sleep()
            # rospy.loginfo(f"waiting for recent available actions, asked for {stamp}, current {self._prompt_stamp}")
        rates = self._rate_actions(self._prompt.body, task, self._prompt.actions)
        response = ActionsRateResponse()
        response.rated_actions = self._prompt
        response.rated_actions.probabilities = rates
        response.done_tasks = self._done_tasks
        return response
    
    def _rate_actions(self, prompt_body, task, actions):
        prompt = prompt_body + "\n" + task + "\n"
        prompt += "\n".join(self._done_tasks) + "\n"
        prompt += "<|endofprompt|>"
        prompt += "<|endofvariant|>".join(actions)
        # rospy.loginfo(f"prompt:\n{prompt}")
        rospy.loginfo(f"checking prompt by {self.model_name} service {openai.api_base}")
        rate = rospy.Rate(0.5)
        while True:
            try:
                completion = openai.Completion.create(model="llama-7B-4b", prompt=prompt, max_tokens=0, logprobs=True, echo=True)
                print(completion)
            except openai.APIError as e:
                rospy.logerr(f"got APIError exception (maybe model not loaded by server?): {e}")
            except openai.error.APIConnectionError as e:
                rospy.logerr(f"got APIConnectionError exception (maybe SimpleAI server not running?): {e}")
            else:
                rospy.loginfo(f"got response by LLM service")
                break
            rate.sleep()
        logprobs_avgs = [sum(choice.logprobs.token_logprobs) / (len(choice.logprobs.token_logprobs)) for choice in completion.choices]
        return logprobs_avgs

    def _add_done_task_cb(self, req):
        rospy.loginfo(f"adding done task: {req.task}")
        self._done_tasks.append(req.task)
        return DoneTaskResponse(self._done_tasks)
    
    def _reset_done_tasks_cb(self, req):
        rospy.loginfo("resetting done tasks")
        self._done_tasks = []
        return EmptyResponse()

if __name__ == "__main__":
    rospy.init_node("llm_service_node")
    description_generator = LLMServiceNode()
    rospy.spin()