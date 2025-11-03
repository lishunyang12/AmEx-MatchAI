## 🏆 AmEx MatchAI – AI-Powered Merchant-Consumer Matchmaking Platform 

Finalist | 2025 American Express GenAI Hackathon <br>
Presented at the American Express Singapore Office 

---

This demo is a **proof-of-concept** from a **consumer's perspective**. It leverages Large Language Models (LLMs) to interpret user intents and match them with suitable merchants based on customer profiles.

**To try the demo for yourself:**<br>
* Add your OpenAI API key to `app.py`.
* Choose an LLM.
* Install the required libraries and run the Streamlit app using the commands below:

```
python -m pip install streamlit pandas openai
python -m streamlit run app.py
```

Reminder to **NEVER** share your API key publicly. This setup is adequate for basic local testing, but if you intend to customize it heavily, 
you can refer to <a href="https://docs.streamlit.io/develop/concepts/connections/secrets-management">Streamlit's secrets management</a> for better security practices.

See `MatchAI.pptx` for a **project overview**.<br>
~~Psst, you might want to use Google Slides so it doesn't look weird because other presentation software deal with tab characters differently.~~

---

### Testing Instructions
1. Select a customer from the drop-down menu on the sidebar.
2. Enter a prompt in the text box (e.g., "Find me a restaurant").
3. Wait for the AI to generate personalized recommendations.
4. (Optional) Refer to the `History` section for more information.

---

### Suggestions for Customization
**Refine Intent**<br>
You can update the content in the parentheses in `refine_intent_with_real_intent()` in `app.py` to influence how the LLM responds.

For example, for `price_range`, how would you categorize a customer based on balance alone? Should a balance of S$9,999 be offered moderate while S\$10,000 be expensive? You get the idea.

Some options include (but are not limited to):
* Define crisp boundaries (Not recommended)
* Define membership values for fuzzy logic
* Ask the LLM to categorize based on relative standing among all customers
* Ask the LLM to compare to real statistics (e.g., median income in Singapore)

**More parameters? More parameters!**<br>
You can also always add more features by explicitly telling the LLM what other parameters the users could provide!

**Choosing the ideal LLM for each step**<br>
For simplicity, our implementation uses the same LLM for all API calls. You can customize this by changing the model name in each of the functions below:
* `extract_intent()`
* `refine_intent_with_real_intent()`
* `deduce_profile_preferences()`
* `generate_personalized_offer()`

**Custom Datasets**<br>
Our current dataset is specifically curated for demonstration purposes. You can try this with your own dataset by replacing `merchants_sg.jsonl` and `customers_sg.jsonl`.