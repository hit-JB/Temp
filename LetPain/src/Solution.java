import org.junit.Test;

import java.util.*;

public class Solution {
    public static void main(String[] args) {
        Solution sol = new Solution();
//        int[] nums = {536870912,0,534710168,330218644,142254206};
//        int[][] queries =  {{558240772,1000000000},{307628050,1000000000},{3319300,1000000000},
//                {2751604,683297522},{214004,404207941}};
//        int[] ret = sol.maximizeXor(nums,queries);
//        for(int e:ret)
//            System.out.println(e);
        int[][] nums = {{2,1},{3,4},{3,2}};
        System.out.println(sol.restoreArray(nums));
    }
    public static void listForeach(ListNode listNode){
        while (listNode!=null) {
            System.out.print(listNode.val + " ");
            listNode = listNode.next;
        }
    }
    public static ListNode createList(int[] nums){
        ListNode head = new ListNode(nums[0]);
        ListNode ret = head;
        for(int i=1;i<nums.length;i++){
            ret.next = new ListNode(nums[i]);
            ret = ret.next;
        }
        return head;
    }
    public ListNode addTwoNumbersII(ListNode l1, ListNode l2) {
        ListNode ret = new ListNode();
        ListNode head = ret;
        int carry = 0;
        while(l1!=null && l2!=null){
            int e1 = l1.val,e2= l2.val;
            int sum = e1+e2 +carry;
            carry = sum>=10?1:0;
            head.next = new ListNode(sum %10);
            head = head.next;
            l1 = l1.next;
            l2=l2.next;
        }
        ListNode list = l1!=null?l1:l2!=null?l2:null;
        if(list==null) {
            if (carry == 1) {
                head.next = new ListNode(1);
                head = head.next;
            }
            head.next = null;
            return ret.next;
        }
        while (list!=null){
            int e =list.val;
            int sum = e+carry;
            carry = sum>=10?1:0;
            head.next = new ListNode(sum %10);
            head = head.next;
            list = list.next;
        }
        if (carry == 1) {
                head.next = new ListNode(1);
                head = head.next;
            }
        head.next = null;
        return ret.next;
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        return 0;
    }
    public double getKElement(int[] nums1,int start1,int end1,
    int[] nums2,int start2,int end2,int k,boolean flag){
        if(k==1) {
            if(!flag)
            return Math.min(nums1[start1], nums2[start2]);
            if(nums1[start1]<=nums1[start1])
            {
                int e2= start1+1>=end1?nums2[start2]:Math.min(nums1[start1+1],nums2[start2]);
                return (e2+nums1[start1]) / 2.;
            }else{
                int e2 = start2+1>=end2?nums1[start1]:Math.min(nums1[start1],nums2[start2+1]);
                return (e2+nums2[start2]) / 2.;
            }
        }
        int e1,e2,det1=0,det2=0;
        if(start1==end1)
            return nums2[start2+k];
        if(start2==end2)
            return nums1[start1+k];
        if (start1+k/2-1>=end1)
        {
            e1 = nums1[end1-1];
            det1 = end1-start1;
        }else{
            e1 = nums1[start1+k/2-1];
            det1=k/2;
        }
        if (start2+k/2-1>=end2)
        {
            e2 = nums2[end2-1];
            det2 = end2-start2;
        }else{
            e2 = nums2[start2+k/2-1];
            det2=k/2;
        }
        if(e1<=e2){
            start1 +=det1;
            k =k-det1;
        }else{
            start2 +=det2;
            k = k-det2;
        }
        return getKElement(nums1,start1,end1,nums2,start2,end2,k,flag);
    }
    public int findIndex(int target,int[] nums){
        if(target < nums[0])
            return 0;
        if(target >= nums[1])
            return nums.length;
        int start = 0,end = nums.length,mid = (start +end) /2;
        while(end-start>1){
            if(nums[mid]>target){
                end = mid;
                mid = start +(end-start) /2;
            }else if(nums[mid]<target){
                start = mid;
                mid = start +(end-start) /2;
            }else{
                return mid;
            }
        }
        return start+1;
    }
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> ret = new ArrayList<>();
        Stack<int[]> stack = new Stack<>();
        int left = newInterval[0],right = newInterval[1];
        int start = -1,end = -1;
        int i=0;
        if(newInterval[1]<intervals[0][0])
            ret.add(newInterval);
        for(;i<intervals.length;i++){
            int[] e = intervals[i];

            if(e[0]<= right && e[0]> left || e[1]<=right && e[1]>=left){
                if(start==-1)
                    start = Math.min(left,e[0]);
                end = Math.max(end,Math.max(right,e[1]));
                if(i==intervals.length-1)
                    ret.add(new int[]{start, end});
                continue;
            }

            if(start!=-1 && end !=-1) {
                ret.add(new int[]{start, end});
                start=-1;
            }
            ret.add(e);
            if(i<intervals.length-1 && e[1]<newInterval[0] && intervals[i+1][0]>newInterval[1])
                ret.add(newInterval);
        }
        if(newInterval[0]>intervals[intervals.length-1][1])
            ret.add(newInterval);
        int[][] retArray = new int[ret.size()][];
        for(int j=0;j<ret.size();j++)
            retArray[j] = ret.get(j);
        return retArray;
    }
    public int findMaximumXOR(int[] nums) {
        int max_size = 30;
        int ret = 0;
        for(int i=max_size;i>=0;i--){
            Set<Integer> seen = new HashSet<>();
            for(int num:nums){
                seen.add(num>>i);
            }
            int x_hot = ret * 2+1;
            boolean found = false;
            for(int num:nums){
                if (seen.contains(x_hot ^ num >> i)) {
                    found = true;
                    break;
                }
            }
            if(found)
                ret = x_hot;
            else
                ret = x_hot-1;
        }
        return ret;
    }
    public int findMaximumXORII(int[] nums){
        NumNode node = new NumNode();
        NumNode head = node;
        int m =Integer.MIN_VALUE;
        for(int e:nums){
            m = Math.max(m,e);
        }
        int pow = (int) (Math.log(m) / Math.log(2));
        for(int e:nums){
            node = head;
            for(int i=pow;i>=0;i--){
                int hot = (e>>i) & 1;
                if(node.next[hot]==null){
                    node.next[hot] = new NumNode();
                }
                node = node.next[hot];
            }
        }
        int max = Integer.MIN_VALUE;
        for(int e:nums){
            int sum = 0;
            node = head;
            for(int i=pow;i>=0;i--){
                int pre = e>>i & 1;
                if(node.next[1-pre]!=null) {
                    sum = 2*sum+1;
                    node = node.next[1 - pre];
                }
                else {
                    node = node.next[pre];
                    sum = 2 *sum;
                }
            }
            max = Math.max(sum,max);
        }
        return max;
    }
    public int[] maximizeXor(int[] nums, int[][] queries) {
        NumNode node = new NumNode();
        NumNode head = node;
        int m =Integer.MIN_VALUE;
        for(int e:nums){
            m = Math.max(m,e);
        }
        int pow = (int) (Math.log(m) / Math.log(2));
        for(int e:nums){
            node = head;
            for(int i=pow;i>=0;i--){
                int hot = (e>>i) & 1;
                if(node.next[hot]==null){
                    node.next[hot] = new NumNode();
                }
                node = node.next[hot];
            }
        }
        int[] ret  = new int[queries.length];
        for(int i=0;i<queries.length;i++){
            int x_i = queries[i][0],m_i = queries[i][1];
            int j = (int)Math.pow(2,pow);
            node = head;
            while(node != null && j>m_i){
                j = j>>1;
                node = node.next[0];
            }
            if(node==null){
                ret[i] = -1;
            }else{
                int x_binary = (int)(Math.log(x_i)/Math.log(2));
                int count=(int)(Math.log(j) / Math.log(2));
                int num = 0;
                for(int k=x_binary;k>count;k--)
                    num += 1<<x_binary;
                NumNode node_ = node;
                while(node_!=null){
                    if(node_.next[1]!=null)
                        break;
                    node_ = node_.next[0];
                }
                if(node_==null)
                    ret[i] =-1;
                else {
                    int max = 0;
                    while (count >= 0) {
                        int e = (x_i >> count) & 1;
                        if (node.next[1 - e] != null) {
                            node = node.next[1 - e];
                            max = 2 * max + 1;
                        } else {
                            node = node.next[e];
                            max = 2 * max;
                        }
                        count--;
                    }
                    ret[i] = num + max;
                }
            }
        }
        return ret;
    }
    public int[] buildArray(int[] nums) {
        int[] ret = new int[nums.length];
        for(int i=0;i<nums.length;i++){
            ret[i] = nums[nums[i]];
        }
        return ret;
    }
    public int eliminateMaximum(int[] dist, int[] speed) {
        List<Integer> time  = new ArrayList<>();
        for(int i=0;i<dist.length;i++) {
            if (dist[i] % speed[i] == 0)
                time.add(dist[i] / speed[i]);
            else
                time.add(dist[i] / speed[i] + 1);
        }
        time.sort(Integer::compare);
        for(int i=0;i<time.size();i++){
            if(time.get(i)<=i)
                return i;
        }
        return time.size();
    }
    public int countGoodNumbers(long n) {
        int mod =(int) Math.pow(10,9) +7;
        long odd,even;
        odd = n/2;
        even = n-odd;
        return (int)((quickPow(4,odd,mod) *
                quickPow(5,even,mod)) % mod);

    }
    public long quickPow(long base,long pow,int mod){
        if (pow==1)
            return base % mod;
        if(pow==0)
            return 1;
        long n = quickPow(base,pow /2,mod);
        if(pow % 2==0){
           return (n*n) % mod;
        }
        return (n * n * base) % mod;
    }
    public int[][] rotateGrid(int[][] grid, int k) {
        int m = grid.length,n = grid[0].length;
        int[][] clockWise = new int[][]{{0,1},{1,0},{0,-1},{-1,0}};
        int[][] unClockWise = new int[][]{{1,0},{0,1},{-1,0},{0,-1}};
        int[][][] direction = new int[][][]{clockWise,unClockWise};
        int i=0,j=0;
        while(m>=2 && n>=2){
            int nums = m * n;
            int k_ = k % nums;
            int rotate = 1;
            if(k_ > nums / 2){
                rotate = 0;
                k_ = nums - k_;
            }
            int row = i,column = j;
            int[][] start_rotate = direction[1-rotate];
            int d=0;
            List<Integer> stack = new ArrayList<>();
            while(k_>=0){
                stack.add(grid[row][column]);
                if(row+start_rotate[d][0] <i || row +start_rotate[d][0] >= i+m ||
                column + start_rotate[d][1] < j || column +start_rotate[d][1] >= j+n)
                    d++;
                row += start_rotate[d][0];column += start_rotate[d][1];
                k_--;
            }
            stack.remove(0);
            Queue<Integer> queue = new ArrayDeque<>();
            while(!stack.isEmpty())
                queue.add(stack.remove(stack.size()-1));
            int[][] num_rotate = direction[rotate];
            d=0;
            row = i;column =j;
            do{
                queue.add(grid[row][column]);
                grid[row][column] = queue.remove();
                if(row+num_rotate[d][0] <i || row +num_rotate[d][0] >=i+m ||
                        column + num_rotate[d][1] < j || column +num_rotate[d][1] >=j+n)
                    d++;
                row += num_rotate[d][0];column+=num_rotate[d][1];
            }while(row!=i || column!=j);
            i++;j++;
            m -=2;n-=2;
        }
        return grid;
    }
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode headA_copy = headA,headB_copy = headB;
        while(headA != headB){
            headA=headA==null?headB_copy:headA.next;
            headB=headB==null?headA_copy:headB.next;
        }
        return headA;
    }
    public long wonderfulSubstrings(String word) {
        Map<Character,Long> map = new HashMap<>();
        Map<Long,Long> count = new HashMap<>();
        count.put(0L, 1L);
        for(char c='a';c<='j';c++){
            map.put(c, 1L <<(c-'a'));
        }
        long ret = 0L;
        long sum=0;
        for(int i=0;i<word.length();i++){
            char c = word.charAt(i);
            sum = sum^map.get(c);
            ret += count.getOrDefault(sum,0L);
            for(int k=0;k<=9;k++)
            {
                ret += count.getOrDefault((1<<k) ^ sum,0L);
            }
            count.put(sum,count.getOrDefault(sum,0L)+1);
        }
        return ret;
    }
    public int waysToBuildRooms(int[] prevRoom) {
        Map<Integer,Integer> construct = new HashMap<>();
        for(int i=1;i<prevRoom.length;i++){
            construct.put(prevRoom[i],i);
        }
        return 0;
    }
    public void dfs(Map<Integer,Set<Integer>> map,Map<Integer,Integer> construct,
                    int[] count,int node){
        if(construct.isEmpty()){
            count[0]++;
            return;
        }

    }
    public Node copyRandomList(Node head) {
        if(head==null)
            return null;
        Node node = new Node(head.val);
        Node ret = node;
        Map<Node,Node> pairNode = new HashMap<>();
        Node start = head;
        pairNode.put(head,node);
        while(start!=null){
            if(pairNode.containsKey(start.next)){
                node.next = pairNode.get(start.next);
            }else{
                node.next = start.next==null?null:new Node(start.next.val);
                pairNode.put(start.next,node.next);
            }

            if(pairNode.containsKey(start.random)){
                node.random = pairNode.get(start.random);
            }else{
                node.random = start.random==null?null:new Node(start.random.val);
                pairNode.put(start.random,node.random);
            }

            start = start.next;
            node = node.next;
        }
        return ret;
    }
    public double largestSumOfAverages(int[] nums, int k) {
        double[][] dp = new double[nums.length][k];
        double[] sum = new double[nums.length];sum[0] = nums[0];
        for(int i=1;i<nums.length;i++)
            sum[i] = sum[i-1] +nums[i];
        for(int i=0;i<nums.length;i++){
            dp[i][0] = sum[i] / (i+1);
            for(int j=1;j<=i && j<k;j++){
                for(int p=0;p<i;p++) {
                    if(p<j-1)
                        continue;
                    dp[i][j] = Math.max(dp[p][j - 1] + (sum[i] - sum[p]) / (i - p), dp[i][j]);
                }
            }
        }
        return dp[dp.length-1][dp[0].length-1];
    }
    public int[][] imageSmoother(int[][] img) {
        int [][] grid = new int[img.length][img[0].length];
        int[][] direction = new int[][]{{0,0},{-1,-1},{-1,0},{-1,1},{0,1},{1,1},
                {1,0},{1,-1},{0,-1}};
        for(int i=0;i<img.length;i++){
            for(int j=0;j<img[0].length;j++){
                int sum=0,count=0;
                for (int[] ints : direction) {
                    int row = i + ints[0], column = j + ints[1];
                    if (row < 0 || row >= img.length ||
                            column < 0 || column >= img[0].length)
                        continue;
                    sum += img[row][column];
                    count += 1;
                }
                grid[i][j] = sum / count;
            }
        }
        return grid;
    }
    public boolean isCovered(int[][] ranges, int left, int right) {
        List<int[]> list = new ArrayList<>(Arrays.asList(ranges));
        list.sort(Comparator.comparingInt(e -> e[0]));
        Stack<int[]> stack = new Stack<>();
        List<Stack<int[]>> ret = new ArrayList<>();
        for(int[] e:list){
            int left_ = e[0],right_ = e[1];
            if(stack.isEmpty()) {
                stack.push(e);
            }
            else{
                if(left_<=stack.peek()[1]+1 && right_>stack.peek()[1])
                    stack.push(e);
                else if(left_>stack.peek()[1]+1){
                    Stack<int[]> temp = new Stack<>();
                    temp.addAll(stack);
                    ret.add(temp);
                    stack.clear();
                    stack.push(e);
                }
            }
        }
        System.out.println(stack);
        ret.add(stack);
        for(Stack<int[]> e:ret){
            if(e.get(0)[0]<=left && e.peek()[1]>= right)
                return true;
        }
        return false;
    }
    public long maxAlternatingSum(int[] nums) {
        if(nums.length==1)
            return nums[0];
        int i=0;
        long sum = 0;
        while(i<nums.length-1){
            while(i+1<nums.length && nums[i+1]>=nums[i])
                i++;
            sum+= nums[i];
            while(i+1<nums.length && nums[i+1]<=nums[i])
                i++;
            if(i+1<nums.length) {
                sum -= nums[i];
            }
        }
        return sum;
    }
    public String maximumTime(String time) {
        StringBuilder builder = new StringBuilder();
        for(int i=0;i<5;i++){
            char c = time.charAt(i);
            if(c=='?'){
                if(i==0)
                {
                    if(time.charAt(i+1)=='?' || time.charAt(i+1)-'0'<4)
                        builder.append(2);
                    else
                        builder.append(1);
                }else
                if(i==1){
                    if(builder.charAt(0)=='2')
                        builder.append(3);
                    else
                        builder.append(9);
                }else
                    if(i==3)
                        builder.append(5);
                else
                        builder.append(9);
            }
            else
                builder.append(c);
        }
        return builder.toString();
    }
    public int maxProfit(int[] prices) {
        int[] buy = new int[prices.length],sale = new int[prices.length];
        buy[0] = -prices[0];
        for(int i=1;i<prices.length;i++){
            sale[i] = Math.max(buy[i-1]+prices[i],sale[i-1]);
            buy[i] = Math.max(buy[i-1],i-2>=0?sale[i-2]-prices[i]:-prices[i]);
        }
        return sale[prices.length-1];
    }
    public int leastBricks(List<List<Integer>> wall) {
        Map<Integer, Integer> count = new HashMap<>();
        for (List<Integer> e : wall) {
            int sum = 0;
            for (int i = 0; i < e.size() - 1; i++) {
                int width = e.get(i);
                sum += width;
                count.put(sum, count.getOrDefault(sum, 0) + 1);
            }
        }
        int max = 0;
        for (Map.Entry<Integer, Integer> e : count.entrySet()) {
            max = Math.max(max, e.getValue());
        }
        return wall.size() - max;
    }
    public int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer,List<Integer>> map = new HashMap<>();
        for(int[] e:adjacentPairs){
            map.computeIfAbsent(e[0],k->new ArrayList<>());
            map.get(e[0]).add(e[1]);
            map.computeIfAbsent(e[1],k->new ArrayList<>());
            map.get(e[1]).add(e[0]);
        }
        int[] ret = new int[adjacentPairs.length+1];
        for(Map.Entry<Integer,List<Integer>> e:map.entrySet()){
            if(e.getValue().size()==1){
                ret[0] = e.getKey();
                break;
            }
        }
        int next ,e;
        for(int i=1;i<ret.length;i++){
            e = ret[i-1];
            List<Integer> list = map.get(e);
            next = list.size()==1 || list.get(0)!= ret[i-2]?list.get(0):list.get(1);
            ret[i] = next;
        }
        return ret;
    }
}
