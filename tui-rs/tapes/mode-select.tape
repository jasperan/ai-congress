Output img/tui-mode-select.gif

Set Shell "bash"
Set FontSize 16
Set Width 1400
Set Height 700
Set Theme "Dracula"
Set TypingSpeed 50ms

Type "cd /home/ubuntu/git/personal/ai-congress && ./tui-rs/target/release/congress-tui --server http://localhost:8000"
Enter
Sleep 3s
Enter
Sleep 2s
# Select models
Down Down Down Down Space
Sleep 300ms
Down Space
Sleep 300ms
# Go to mode select
Enter
Sleep 2s
Screenshot img/tui-mode-select.png
# Switch to simulation mode
Right
Sleep 1s
Screenshot img/tui-mode-select-sim.png
Sleep 1s
