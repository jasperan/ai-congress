Output img/tui-simulation.gif

Set Shell "bash"
Set FontSize 14
Set Width 1400
Set Height 700
Set Theme "Dracula"
Set TypingSpeed 50ms

Type "cd /home/ubuntu/git/personal/ai-congress && ./tui-rs/target/release/congress-tui --simulation --topic 'Should AI systems be regulated by federal law?' --agents 5 --ticks 20 --model smollm2:135m --server http://localhost:8000"
Enter
Sleep 30s
Screenshot img/tui-simulation.png
Sleep 5s
Screenshot img/tui-simulation-2.png
Sleep 1s
