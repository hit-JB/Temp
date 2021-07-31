import java.util.*;

public class Solution {
    public static void main(String[] args) {
        Solution sol = new Solution();
        TreeNode root = new TreeNode(3,new TreeNode(9),new TreeNode(20,new TreeNode(15),new TreeNode(7)));
        System.out.println(sol.verticalTraversal(root));
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
    public Node_ copyRandomList(Node_ head) {
        if(head==null)
            return null;
        Node_ node = new Node_(head.val);
        Node_ ret = node;
        Map<Node_,Node_> pairNode = new HashMap<>();
        Node_ start = head;
        pairNode.put(head,node);
        while(start!=null){
            if(pairNode.containsKey(start.next)){
                node.next = pairNode.get(start.next);
            }else{
                node.next = start.next==null?null:new Node_(start.next.val);
                pairNode.put(start.next,node.next);
            }

            if(pairNode.containsKey(start.random)){
                node.random = pairNode.get(start.random);
            }else{
                node.random = start.random==null?null:new Node_(start.random.val);
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
    public int longestSubarray(int[] nums) {
        return 0;
    }
    public boolean pyramidTransition(String bottom, List<String> allowed) {
        Map<String,List<Character>> map = new HashMap<>();
        for(String e:allowed){
            map.computeIfAbsent(e.substring(0, 2), k -> new ArrayList<>());
            map.get(e.substring(0,2)).add(e.charAt(2));
        }
        return dfsPyramid(map,bottom);
    }
    public boolean dfsPyramid(Map<String,List<Character>> map,String bottom){
        if(bottom.length()==1)
            return true;
        boolean is;
        List<String> ret = new ArrayList<>();
        getNextFloor(map,bottom,"",0,ret);
        if(ret.size()==0)
            return false;
        for(String e:ret){
            is = dfsPyramid(map,e);
            if(is)return true;
        }
        return false;
    }
    public void getNextFloor(Map<String,List<Character>> map,
                             String button, String s,int i,List<String> ret
                            ){

        if(i==button.length()-1)
        {
            ret.add(s);
            return;
        }
        String e = button.substring(i,i+2);
        List<Character> list = map.getOrDefault(e,null);
        if(list==null) {
            return;
        }
        for(Character c:list)
        {
            getNextFloor(map,button,s+c,i+1,ret);
        }
    }
    public int minOperations(int[] target, int[] arr) {
        TreeSet<Integer> tree = new TreeSet<>();
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<target.length;i++){
            map.put(target[i],i);
        }
        int[] local = new int[arr.length];
        for(int i=0;i<arr.length;i++) {
            Integer loc = map.getOrDefault(arr[i], null) ;
            local[i] = loc==null?-1:loc;
        }
        for (int j : local) {
            if (j == -1)
                continue;
            if (tree.isEmpty() || j > tree.last()) {
            }
            else {
                Integer e = tree.floor(j);
                if (e != null) {
                    tree.remove(e);
                }
            }
            tree.add(j);
        }
        return target.length-tree.size();
    }
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length()][text2.length()];
        for(int i=0;i<dp.length;i++){
            dp[i][0] = text1.substring(0,i+1).indexOf(text2.charAt(0))==-1?0:1;
            for(int j=1;j<dp[0].length;j++){
                if(i==0)
                    dp[i][j] = text2.substring(0,j+1).indexOf(text1.charAt(0))==-1?0:1;
                else{
                    int loc = text1.substring(0,i+1).lastIndexOf(text2.charAt(j));
                    dp[i][j] =dp[i][j-1];
                   if(loc!=-1){
                       if(loc==0) {
                           dp[i][j] = Math.max(dp[i][j],1);
                       }
                       else
                        dp[i][j] = Math.max(dp[i][j],dp[loc-1][j-1]+1);
                   }
                }
            }
        }
        return dp[dp.length-1][dp[0].length-1];
    }
    public int findSecondMinimumValue(TreeNode root) {
        int[] min= findSecondAndFirst(root);
        return min[1];
    }
    public int[] findSecondAndFirst(TreeNode root){
        if(root.left==null){
            return new int[]{root.val,root.val};
        }
        int[] left = findSecondAndFirst(root.left),
                right = findSecondAndFirst(root.right);
        Set<Integer> set = new HashSet<>();
        for(int e:left)
            set.add(e);
        for(int e:right)
            set.add(e);
        set.add(root.val);
        List<Integer> list = new ArrayList<>(set);
        list.sort(Integer::compare);
        if(list.size()==1)
            return new int[]{list.get(0),list.get(0)};
        return new int[]{list.get(0),list.get(1)};
    }
    public int eatenApples(int[] apples, int[] days) {
        PriorityQueue<int[]> queue = new PriorityQueue<>(Comparator.comparingInt(e -> e[0]));
        int sum = 0;
        int day = 0;
        do
        {
            if(day<apples.length && apples[day]!=0)
                 queue.add(new int[]{day+days[day],apples[day]});
            int[] e= queue.peek();
            if(queue.isEmpty() && day<apples.length)
            {
                day++;continue;
            }
            if(e[0] > day && e[1]>0){
                e[1]--;
                sum++;
                day++;
            }else{
                queue.remove();
            }
        }while(!queue.isEmpty() || day < apples.length);
        return sum;
    }
    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> treeNodes = new ArrayDeque<>();
        TreeNode bottom_left = root;
        treeNodes.add(root);
        while(!treeNodes.isEmpty()){
            int size = treeNodes.size();
            bottom_left = treeNodes.peek();
            for(int i=0;i<size;i++){
                TreeNode node = treeNodes.remove();
                if(node.left!=null)
                    treeNodes.add(node.left);
                if(node.right!=null)
                    treeNodes.add(node.right);
            }
        }
        assert bottom_left != null;
        return bottom_left.val;
    }
    public int numWays(String s) {
        List<Integer> loc = new ArrayList<>();
        for(int i=0;i<s.length();i++){
            if(s.charAt(i)=='1')
                loc.add(i);
        }
        int size = loc.size();
        if(size % 3!=0)
            return 0;
        long mod = (long)10e9+7;
        if(size==0)
        {
            long length = s.length()-1;
            return (int)((length * (length-1)  / 2) % mod);
        }
        long index1 = loc.get(size / 3-1),index2 = loc.get(size / 3),
                index3 = loc.get(size / 3  *2-1),index4 = loc.get(size / 3 * 2);
        return (int) (((index2-index1) * (index4-index3)) % mod );
    }
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        TreeNodeWithParent[] start = new TreeNodeWithParent[1];
        convert(null,new TreeNodeWithParent(root.val),root,target.val,start);
        Queue<TreeNodeWithParent> queue = new ArrayDeque<>();
        queue.add(start[0]);
        List<Integer> ret = new ArrayList<>();
        if(k==0)
        {
            ret.add(start[0].val);
            return ret;
        }
        Set<TreeNodeWithParent> visited = new HashSet<>();
        visited.add(start[0]);
        int step = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            step++;
            for(int i=0;i<size;i++){
                TreeNodeWithParent top = queue.poll();
                if(top.left!=null && !visited.contains(top.left))
                {
                    queue.add(top.left);
                    visited.add(top.left);
                }
                if(top.right!=null && !visited.contains(top.right))
                {
                    queue.add(top.right);
                    visited.add(top.right);
                }
                if(top.parent!=null && !visited.contains(top.parent))
                {
                    queue.add(top.parent);
                    visited.add(top.parent);
                }
            }
            if(step==k)
            {
                for(TreeNodeWithParent e:queue)
                    ret.add(e.val);
                return ret;
            }
        }
        return null;
    }
    public void convert(TreeNodeWithParent parent,TreeNodeWithParent convertNode,
                        TreeNode root, int target,TreeNodeWithParent[] start){
        if(root.left!=null)
        {
            convertNode.left = new TreeNodeWithParent(root.left.val);
            convert(convertNode,convertNode.left,root.left,target,start);
        }
        convertNode.parent = parent;
        if(convertNode.val==target)
            start[0] = convertNode;
        if(root.right!=null){
            convertNode.right = new TreeNodeWithParent(root.right.val);
            convert(convertNode,convertNode.right,root.right,target,start);
        }
    }
    public static class TreeNodeWithParent{
         TreeNodeWithParent parent;
         TreeNodeWithParent left;
         TreeNodeWithParent right;
         int val;
        public TreeNodeWithParent(int val){
            this.val = val;
        }
    }
    public int chalkReplacer(int[] chalk, int k) {
        int length = chalk.length;
        long[] sum = new long[chalk.length];
        sum[0] = chalk[0];
        for(int i=1;i<length;i++){
            sum[i] = sum[i-1]+chalk[i];
        }
        long count = k / sum[length-1];
        long remain = k - count * sum[length-1];
        int loc = divFind(sum,remain);
        return loc;
    }
    public int divFind(long[] num,long target){
        int low = 0,high = num.length,
                mid = low + (high-low) / 2;
        while(high-low>1){
            if(num[mid]>target){
                high = mid;
                mid = low +(high-low) / 2;
            }else if(num[mid]<target){
                low = mid;
                mid = low +(high-low) / 2;
            }else{
                return mid+1;
            }
        }
        if(num[low]<=target)
            return low+1;
        else
            return 0;
    }
    public int largestMagicSquare(int[][] grid) {
        int[][] row_sum = new int[grid.length][grid[0].length];
        int[][] column_sum = new int[grid.length][grid[0].length];
        int[][] lDiagSum= new int[grid.length][grid[0].length],
                rDiagSum = new int[grid.length][grid[0].length];
        for(int i=0;i< grid.length;i++){
            for(int j=0;j< grid[0].length;j++){
                row_sum[i][j] = j-1>=0?row_sum[i][j-1]+grid[i][j]:0;
                column_sum[i][j] = i-1>=0?column_sum[i-1][j]+grid[i][j]:0;
                lDiagSum[i][j] = i-1>=0 && j-1>=0?lDiagSum[i-1][j-1]+grid[i][j]:grid[i][j];
            }
        }
        for(int i=0;i< grid.length;i++){
            for(int j=grid[0].length-1;j>=0;j--)
                rDiagSum[i][j] = i-1>=0 && j+1<grid[0].length?rDiagSum[i-1][j+1]+grid[i][j]:
                        grid[i][j];
        }

        int max_size = Math.min(grid.length,grid[0].length);
        for(int k=max_size;k>1;k--){
            for(int i=0;i+k<= grid.length;i++){
                for(int j=0;j+k<=grid[0].length;j++){
                    if(checkGrid(i,j,k,row_sum,column_sum,lDiagSum,rDiagSum))
                        return k;
                }
            }
        }
        return 1;
    }
    public boolean checkGrid(int i,int j,int k,int[][] row_sum,int[][] column_sum,
                             int[][] lDiag,int [][] rDiag){
        int sum = row_sum[i][j+k-1] - (j>0?row_sum[i][j-1]:0);
        int sum_ = column_sum[i][j+k-1] - (i>0?row_sum[i-1][j]:0);
        int lSum = lDiag[i+k-1][j+k-1] - (i-1>=0&&j-1>=0?lDiag[i-1][j-1]:0);
        int rSum = rDiag[i+k-1][j] - (i-1>0&&j+k<rDiag[0].length?rDiag[i-1][j+k]:0);
        if(sum!=sum_ || sum!=lSum||sum!=rSum)
            return false;
        for(int p=0;p<k;p++){
            if(row_sum[i+p][j+k-1]-(j>0?row_sum[i+p][j-1]:0) != sum)
                return false;
            if(column_sum[i+k-1][j+p] -(i>0?row_sum[i-1][j]:0) !=sum_)
                return false;
        }
        return true;
    }
    public List<Integer> pathInZigZagTree(int label) {
        int n = (int)(Math.log(label+1) / Math.log(2));
        int remain = (int) (label- Math.pow(2,n)+1);
        int row_label;
        int start_floor;
        if(remain ==0){
            if(n+1 % 2==1)
                row_label = label;
            else
                row_label = (int) (Math.pow(2,n-1));
            start_floor = n-1;
        }else{
            start_floor = n;
            if(n+1 % 2==1)
                row_label = label;
            else
                row_label = (int)(Math.pow(2,n) + Math.pow(2,n) - remain);
        }
        List<Integer> ret = new ArrayList<>();
        while(row_label>=1){
            int index;
            if(start_floor+1 % 2==1)
            {
                index = row_label;
            }else{
                index = (int)(Math.pow(2,start_floor+1)-row_label+Math.pow(2,start_floor)-1);
            }
            ret.add(index);
            row_label = row_label / 2;
            start_floor --;
        }
        Collections.reverse(ret);
        return ret;
    }
    public int minimumTeachings(int n, int[][] languages, int[][] friendships) {
        Set<Integer> foreign = new HashSet<>();
        for(int[] e:friendships){
            if(!canCall(languages[e[0]-1],languages[e[1]-1])) {
                foreign.add(e[0]);
                foreign.add(e[1]);
            }
        }
        int[] l = new int[n];
        for(Integer e:foreign){
            for(int language:languages[e-1])
                l[language]++;
        }
        int max = 0;
        for(int e:l)
            max = Math.max(e,max);
        return foreign.size() - max;
    }
    public boolean canCall(int[] e1,int[] e2){
        Set<Integer> set = new HashSet<>();
        for(int e:e1)
            set.add(e);
        for(int e:e2) {
            if (set.contains(e))
                return true;
        }
        return false;
    }
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        List<Node>  nodes = new ArrayList<>();
        dfsVertical(root,0,0,nodes);
        nodes.sort(Node::compareTo);
        List<List<Integer>> ret = new ArrayList<>();
        int i = 0;
        while(i<nodes.size()){
            List<Integer> temp = new ArrayList<>();
            while(i<nodes.size()-1 && nodes.get(i+1).loc[1]==nodes.get(i).loc[1])
            {
                temp.add(nodes.get(i).val);
                i++;
            }
            temp.add(nodes.get(i).val);
            i++;
            ret.add(new ArrayList<>(temp));
        }
        return ret;
    }
    public void dfsVertical(TreeNode root,int row,int column,List<Node> nodes){
        if(root==null)
            return;
        nodes.add(new Node(new int[]{row,column},root.val));
        dfsVertical(root.left,row+1,column-1,nodes);
        dfsVertical(root.right,row+1,column+1,nodes);
    }
    public static class Node implements Comparable<Node>{
        int[] loc;
        int  val;
        public Node(int[] loc,int val){
            this.loc = loc;
            this.val = val;
        }
        @Override
        public int compareTo(Node o) {
            if(o.loc[0]==this.loc[0] && this.loc[1]== o.loc[1])
                return Integer.compare(this.val,o.val);
             else if(o.loc[1]==this.loc[1])
             return Integer.compare(this.loc[0],o.loc[0]);
             else return Integer.compare(this.loc[1],o.loc[1]);
        }
    }
}

