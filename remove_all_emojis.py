import re
import os

# Comprehensive emoji pattern
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U0001FA00-\U0001FA6F"  # chess symbols
        u"\U0001FA70-\U0001FAFF"  # symbols extended
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        u"\U0000FE00-\U0000FE0F"  # variation selectors
        u"\U0001F018-\U0001F270"  # asian characters
        u"\U0001F200-\U0001F251"
        "]+", flags=re.UNICODE)

# List of Python files to clean
python_files = [
    'app.py',
    'backend_api.py',
    'social_media_detox_analyzer.py',
    'social_media_detox_analyzer_core.py',
    'run_complete_analysis.py'
]

files_processed = 0
emojis_removed = 0

for filename in python_files:
    if os.path.exists(filename):
        print(f"\nProcessing {filename}...")
        
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count emojis before
        before_count = len(emoji_pattern.findall(content))
        
        # Remove emojis
        content_clean = emoji_pattern.sub('', content)
        
        # Also remove specific emoji characters that might be missed
        specific_emojis = ['🧠', '📱', '🔴', '❓', '🏠', '🤖', '⏱️', '🔮', '📊', '📈', 
                          '💡', '🚀', '📝', '✏️', '🎯', '💾', '😴', '📉', '👋', '😊', 
                          '💪', '📥', '🎈', '🎉', '🛈', '🌙', '😢', '😐', '🟢', '🟡', 
                          '🔴', '🚨', '⚠️', '🏃', '✨', '🎨', '🌈', '💎', '🔥', '⚡', 
                          '💫', '🎊', '🛠️']
        
        for emoji in specific_emojis:
            content_clean = content_clean.replace(emoji, '')
        
        # Count emojis after
        after_count = len(emoji_pattern.findall(content_clean))
        removed = before_count - after_count
        
        # Write back
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content_clean)
        
        print(f"  ✓ Removed {removed} emojis from {filename}")
        files_processed += 1
        emojis_removed += removed
    else:
        print(f"  ✗ {filename} not found")

print(f"\n{'='*50}")
print(f"SUMMARY:")
print(f"Files processed: {files_processed}")
print(f"Total emojis removed: {emojis_removed}")
print(f"{'='*50}")
print("\n✓ All emojis removed successfully!")
