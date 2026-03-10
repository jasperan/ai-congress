package components

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jasperan/ai-congress/tui/internal/theme"
)

const (
	bucketCount    = 60
	sparkChars     = "▁▂▃▄▅▆▇█"
	rateWindowSecs = 5
)

var sparkRunes = []rune(sparkChars)

type ThroughputTracker struct {
	mu      sync.Mutex
	buckets [bucketCount]int
	times   [bucketCount]int64
	head    int
}

func NewThroughputTracker() *ThroughputTracker {
	return &ThroughputTracker{}
}

func (t *ThroughputTracker) Add(tokenCount int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now().Unix()
	t.advance(now)
	t.buckets[t.head] += tokenCount
}

func (t *ThroughputTracker) advance(now int64) {
	if t.times[t.head] == now {
		return
	}
	if t.times[t.head] == 0 {
		t.times[t.head] = now
		return
	}
	gap := now - t.times[t.head]
	if gap <= 0 {
		return
	}
	if gap > bucketCount {
		gap = bucketCount
	}
	for i := int64(0); i < gap; i++ {
		t.head = (t.head + 1) % bucketCount
		t.buckets[t.head] = 0
		t.times[t.head] = now - gap + i + 1
	}
}

func (t *ThroughputTracker) CurrentRate() float64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now().Unix()
	t.advance(now)
	var total int
	var count int
	for i := 0; i < rateWindowSecs; i++ {
		idx := (t.head - i + bucketCount) % bucketCount
		if t.times[idx] > 0 && now-t.times[idx] < rateWindowSecs {
			total += t.buckets[idx]
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return float64(total) / float64(count)
}

func (t *ThroughputTracker) Sparkline(width int) string {
	t.mu.Lock()
	defer t.mu.Unlock()
	now := time.Now().Unix()
	t.advance(now)
	if width <= 0 {
		return ""
	}
	if width > bucketCount {
		width = bucketCount
	}
	values := make([]int, width)
	maxVal := 1
	for i := 0; i < width; i++ {
		idx := (t.head - width + 1 + i + bucketCount) % bucketCount
		values[i] = t.buckets[idx]
		if values[i] > maxVal {
			maxVal = values[i]
		}
	}
	var sb strings.Builder
	numChars := len(sparkRunes)
	for _, v := range values {
		charIdx := v * (numChars - 1) / maxVal
		if charIdx >= numChars {
			charIdx = numChars - 1
		}
		sb.WriteRune(sparkRunes[charIdx])
	}
	return sb.String()
}

func RenderThroughput(t *ThroughputTracker, width int) string {
	rate := t.CurrentRate()
	sparkWidth := width - 20
	if sparkWidth < 5 {
		sparkWidth = 5
	}
	spark := t.Sparkline(sparkWidth)
	rateStr := theme.Throughput.Render(fmt.Sprintf("%.1f tok/s", rate))
	return fmt.Sprintf("%s %s", spark, rateStr)
}
