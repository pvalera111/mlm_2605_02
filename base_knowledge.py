rhel_expert_matrix = [
    #FIREWALL-CMD Варіант A: Відкриття порту
    {
        "intent": "open permanent port service traffic rule configuration firewall cmd reload reload",
        "solution": "firewall-cmd --permanent --add-port=8080/tcp && firewall-cmd --reload"
    },
    # Варіант B: Повний аудит поточних правил
    {
        "intent": "list all active firewall rules services ports rich policies zone configuration current status query",
        "solution": "firewall-cmd --list-all"
    },
    # Варіант C: Аварійне блокування всього трафіку (Panic Mode)
    {
        "intent": "emergency isolate network block all inbound outbound connections active panic mode drop firewall enable",
        "solution": "firewall-cmd --panic-on"
    },
    # Варіант D: Створення Rich Rule для блокування конкретного IP
    {
        "intent": "create rich rule reject drop traffic specific rogue bad ip address source block firewall blacklist",
        "solution": "firewall-cmd --permanent --add-rich-rule='rule family=\"ipv4\" source address=\"10.0.0.5\" drop' && firewall-cmd --reload"
    },
    #TCPDUMP  Варіант A: Швидкий базовий сніффінг у терміналі
    {
        "intent": "capture analyze network packets traffic dump real time interface headers sniffing view wire packets",
        "solution": "tcpdump -i any -n"
    },
    # Варіант B: Фільтрація за конкретним хостом та портом
    {
        "intent": "filter network traffic restrict specific target destination source ip address port number tcpdump tcp",
        "solution": "tcpdump -i any -n host 192.168.1.50 and port 443"
    },
    # Варіант C: Запис сирого трафіку у файл (PCAP)
    {
        "intent": "write capture dump raw network packets binary pcap file future analysis trace wire storage",
        "solution": "tcpdump -i any -w /tmp/capture.pcap"
    },
    # Варіант D: Читання раніше записаного PCAP файлу
    {
        "intent": "read analyze decode saved pcap packet capture file ascii hex parsing formatted output tcpdump",
        "solution": "tcpdump -nnnn -r /tmp/capture.pcap"
    },

    # --- КЕРУВАННЯ КОНТЕЙНЕРАМИ PODMAN ---
    {
        "intent": "instantiate launch spawn spinup deploy run isolated container server background map internal external ports dynamic persistent storage volumes mount directory restart policy podman nginx application instance",
        "solution": "podman run -d -p 8080:80 -v /opt/data:/data:Z --restart always --name rhel_app nginx"
    },
    {
        "intent": "list inventory show active running container images processes global resource utilization inspect microservices status pods architecture tree layout overview tracking system metadata queries",
        "solution": "podman ps -a --pod"
    },
    {
        "intent": "enter log execute shell interactive terminal session boundary attach into running active background container namespace spawn bash shell access exist inner commandline inside diagnostic live environment",
        "solution": "podman exec -it rhel_app /bin/bash"
    },
    # --- МЕРЕЖА, ПОРТИ ТА ФАЙРВОЛЛ ---
    {
        "intent": "check network interface status link speed statistics layout ip addressing configuration print info",
        "solution": "ip a"
    },
    {
        "intent": "show arp cache table lookup look active ip addresses and mac tables links for network interface",
        "solution": "ip neigh"
    },
    {
        "intent": "monitor active internet connections open ports listening tcp udp sockets",
        "solution": "ss -tulpn"
    },

    # --- ВІДЛАДКА ЯДРА (KPROBES & EBPF) ---
    {
        "intent": "trace kernel functions events dynamic debugging probes kprobes kretprobes runtime",
        "solution": "echo 'p:myprobe do_sys_open' >> /sys/kernel/debug/tracing/kprobe_events"
    },
    {
        "intent": "monitor system calls kernel execution tracing ebpf tools bpftrace script bpf",
        "solution": "bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'"
    },

    # --- ФОНОВИЙ ЗАПУСК ТА СЕСІЇ (NOHUP, SCREEN, TMUX) ---
    {
        "intent": "run command background survive logout session terminal close standard input output redirection nohup",
        "solution": "nohup python3 script.py > output.log 2>&1 &"
    },
    {
        "intent": "start terminal multiplexer session background split screen layout persistent detach attach tmux",
        "solution": "tmux new -s expert_session"
    },
    {
        "intent": "create persistent virtual terminal console remote management screen multiplexing tool detach",
        "solution": "screen -S system_run"
    },

    # --- ЧИТАННЯ ФАЙЛІВ ТА ПЕРЕГЛЯД ДИСКІВ ---
    {
        "intent": "stream read view text file contents screen page interactive navigation scrolling less",
        "solution": "less logfile.txt"
    },
    {
        "intent": "print display text file lines beginning header start preview head",
        "solution": "head -n 20 file.txt"
    },
    {
        "intent": "monitor real time log updates append lines watch tail follow continuous output",
        "solution": "tail -f /var/log/messages"
    },
    {
        "intent": "inline text stream stream editor search replace regex find pattern modify file sed",
        "solution": "sed -i 's/old_value/new_value/g' config.conf"
    },
    {
        "intent": "list block devices storage drive partitions layout size volume tree diagram lsblk",
        "solution": "lsblk"
    },

    # --- СИСТЕМНІ МЕТРИКИ ТА ПІДСИСТЕМА /PROC ---
    {
        "intent": "dump kernel memory utilization layout physical hardware statistics information data proc meminfo",
        "solution": "cat /proc/meminfo"
    },
    {
        "intent": "inspect central processing unit cpu cores architecture vendor topology inside proc system cpuinfo",
        "solution": "cat /proc/cpuinfo"
    },

    # --- КЕРУВАННЯ КОНТЕЙНЕРАМИ PODMAN ---
    {
        "intent": "run isolated container in background map ports persistent storage volumes restart policy podman",
        "solution": "podman run -d -p 8080:80 -v /opt/data:/data:Z --restart always --name rhel_app nginx"
    },
    {
        "intent": "list active running container images resource utilization inspect microservices pods layout",
        "solution": "podman ps -a --pod"
    },

    # --- БЕЗПЕКА ТА НАЛАШТУВАННЯ SELINUX ---
    {
        "intent": "check current selinux enforcement status mode security policy operational state configuration",
        "solution": "getenforce"
    },
    {
        "intent": "change selinux context permanently restore default security tags file system directory tree restorecon",
        "solution": "semanage fcontext -a -t httpd_sys_content_t '/web(/.*)?' && restorecon -R -v /web"
    }
]
