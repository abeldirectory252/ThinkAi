---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_57/3077170989.py in <cell line: 0>()
     18 ]
     19 
---> 20 inputs = processor.apply_chat_template(
     21     messages,
     22     add_generation_prompt=True,

/usr/local/lib/python3.12/dist-packages/transformers/processing_utils.py in apply_chat_template(self, conversation, chat_template, **kwargs)
   1674                 chat_template = self.chat_template
   1675             else:
-> 1676                 raise ValueError(
   1677                     "Cannot use apply_chat_template because this processor does not have a chat template."
   1678                 )

ValueError: Cannot use apply_chat_template because this processor does not have a chat template.