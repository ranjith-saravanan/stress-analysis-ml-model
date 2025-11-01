import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Comprehensive emoji pattern - covers ALL unicode emoji ranges
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        u"\U0001FA00-\U0001FA6F"  # chess symbols
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        u"\U0000FE00-\U0000FE0F"  # variation selectors
        u"\U0001F018-\U0001F270"  # various asian characters
        u"\U0001F200-\U0001F251"
        "]+", flags=re.UNICODE)

content_no_emoji = emoji_pattern.sub('', content)

# Also remove common text emojis that might have been missed
text_emojis_to_remove = [
    '🧠', '📱', '🔴', '❓', '🏠', '🤖', '⏱️', '🔮', '📊', '📈', '💡',
    '🚀', '📝', '✏️', '🎯', '💾', '😴', '📉', '👋', '😊', '💪', '📥',
    '🎈', '🎉', '🛈', '🌙', '😢', '😐', '🟢', '🟡', '🔴', '🚨', '⚠️', 
    '🏃', '💪', '✨', '🎨', '🌈', '💎', '🔥', '⚡', '💫', '🎊'
]

for emoji in text_emojis_to_remove:
    content_no_emoji = content_no_emoji.replace(emoji, '')

# Clean up any double spaces or weird spacing
content_no_emoji = re.sub(r'  +', ' ', content_no_emoji)
content_no_emoji = re.sub(r'^ ', '', content_no_emoji, flags=re.MULTILINE)

# Save to file
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content_no_emoji)

print("All emojis removed from app.py successfully!")
